"""Minimal Moondream 3.1 inference for ``detect``, with MoE experts offloaded to RAM.
Here the dense part (attention, vision, region head: ~1.6 GB) stays in VRAM,
as many MoE layers as the VRAM budget allows stay there too, and the rest of the fp8 experts
live in pinned RAM. Those are copied into a small GPU pool only when the router picks them.

Architecture and math follow the Moondream 3 reference implementation
(https://huggingface.co/moondream/moondream3-preview), whose code is released under the
Business Source License 1.1 by M87 Labs; the image cropping helpers below are adapted from it.
The 3.1 checkpoint layout follows kestrel, Moondream's inference engine. The model weights are
downloaded at runtime and are covered by the Moondream Model License
(https://moondream.ai/licenses/model/1.0).
"""

import logging
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

HF_REPO = "moondream/moondream3.1-9B-A2B"
HF_REVISION = "1da27e6c2d2f9118de14560df456881dc62d91f3"
TOKENIZER_REPO = "moondream/starmie-v1"

# Text model (3.1 uses the Moondream 3 configuration).
DIM = 2048
N_LAYERS = 24
N_HEADS = 32
HEAD_DIM = DIM // N_HEADS
MOE_START = 4
N_EXPERTS = 64
TOP_K = 8
EXPERT_INNER = 1024
ROPE_THETA = 1_500_000.0
ROT_DIM = 32

# Vision encoder.
ENC_DIM = 1152
ENC_LAYERS = 27
ENC_HEADS = 16
PATCH = 14
CROP = 378
GRID = CROP // PATCH  # 27
MAX_CROPS = 12
OVERLAP = 4
IMAGE_PREFIX = 1 + GRID * GRID  # BOS + 729 image tokens, attended bidirectionally

# Tokens.
BOS_ID = 0
EOS_ID = 0
COORD_ID = 5
DETECT_PREFIX = [1, 7235, 476, 2]
DETECT_SUFFIX = [3]

UP_BYTES = 2 * EXPERT_INNER * DIM  # fp8 bytes of one expert's up projection
DOWN_BYTES = DIM * EXPERT_INNER
EXPERT_BYTES = UP_BYTES + DOWN_BYTES
# Room left for activations (vision crops, prefill attention, dequantized experts).
ACTIVATION_HEADROOM_BYTES = int(1.2e9)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class _Linear:
    __slots__ = ("weight", "bias")

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        self.weight, self.bias = weight, bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class _LayerNorm:
    __slots__ = ("weight", "bias")

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        self.weight, self.bias = weight, bias

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias)


# ---------------------------------------------------------------------------
# Image crops (overlap-and-resize tiling of the reference implementation)
# ---------------------------------------------------------------------------


def _select_tiling(height: int, width: int, crop_size: int, max_crops: int) -> Tuple[int, int]:
    if height <= crop_size or width <= crop_size:
        return (1, 1)
    min_h = math.ceil(height / crop_size)
    min_w = math.ceil(width / crop_size)
    if min_h * min_w > max_crops:
        ratio = math.sqrt(max_crops / (min_h * min_w))
        return (max(1, math.floor(min_h * ratio)), max(1, math.floor(min_w * ratio)))
    h_tiles = max(math.floor(math.sqrt(max_crops * height / width)), min_h)
    w_tiles = max(math.floor(math.sqrt(max_crops * width / height)), min_w)
    if h_tiles * w_tiles > max_crops:
        if w_tiles > h_tiles:
            w_tiles = math.floor(max_crops / h_tiles)
        else:
            h_tiles = math.floor(max_crops / w_tiles)
    return (max(1, h_tiles), max(1, w_tiles))


def _overlap_crops(image: np.ndarray, max_crops: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Global crop at index 0, then overlapping full-resolution tiles."""
    height, width = image.shape[:2]
    margin = PATCH * OVERLAP
    window = (GRID - 2 * OVERLAP) * PATCH
    tiling = _select_tiling(height - 2 * margin, width - 2 * margin, window, max_crops)

    crops = np.zeros((tiling[0] * tiling[1] + 1, CROP, CROP, 3), dtype=np.uint8)
    target_h = tiling[0] * window + 2 * margin
    target_w = tiling[1] * window + 2 * margin
    pil = Image.fromarray(image)
    resized = np.asarray(pil.resize((target_w, target_h), resample=Image.Resampling.LANCZOS))
    crops[0] = np.asarray(pil.resize((CROP, CROP), resample=Image.Resampling.LANCZOS))
    for i in range(tiling[0]):
        for j in range(tiling[1]):
            y0, x0 = i * window, j * window
            region = resized[y0 : min(y0 + CROP, target_h), x0 : min(x0 + CROP, target_w)]
            crops[1 + i * tiling[1] + j, : region.shape[0], : region.shape[1]] = region
    return crops, tiling


def _reconstruct(local: torch.Tensor, tiling: Tuple[int, int]) -> torch.Tensor:
    """Stitch per-tile patch features (``[n, 27, 27, C]``) without their overlap margins."""
    tiles_h, tiles_w = tiling
    size = local.shape[1]
    inner = size - 2 * OVERLAP
    out = local.new_zeros(inner * tiles_h + 2 * OVERLAP, inner * tiles_w + 2 * OVERLAP, local.shape[-1])
    for idx in range(local.shape[0]):
        ty, tx = divmod(idx, tiles_w)
        x0 = 0 if tx == 0 else OVERLAP
        x1 = size if tx == tiles_w - 1 else size - OVERLAP
        y0 = 0 if ty == 0 else OVERLAP
        y1 = size if ty == tiles_h - 1 else size - OVERLAP
        oy, ox = ty * inner, tx * inner
        out[oy + y0 : oy + y1, ox + x0 : ox + x1] = local[idx, y0:y1, x0:x1]
    return out


# ---------------------------------------------------------------------------
# Expert storage
# ---------------------------------------------------------------------------


class ExpertStore:
    """
    fp8 MoE experts split between VRAM and pinned RAM.

    Resident layers keep all 64 experts on the GPU. Offloaded layers live in pinned RAM
    and are copied on demand into a pool of ``N_EXPERTS`` GPU slots, evicted least recently
    used, so a prefill (which touches every expert of a layer) always fits.
    """

    def __init__(self, n_resident_layers: int, device: torch.device):
        self.device = device
        self.moe_layers = list(range(MOE_START, N_LAYERS))
        # Offload the earliest MoE layers: any choice works, this one is deterministic.
        n_offloaded = len(self.moe_layers) - n_resident_layers
        self.offloaded = set(self.moe_layers[:n_offloaded])

        self.up_scale: Dict[int, torch.Tensor] = {}
        self.down_scale: Dict[int, torch.Tensor] = {}
        self.resident_up: Dict[int, torch.Tensor] = {}
        self.resident_down: Dict[int, torch.Tensor] = {}
        self.host_up: Dict[int, torch.Tensor] = {}
        self.host_down: Dict[int, torch.Tensor] = {}

        self.pool_up = self.pool_down = None
        if self.offloaded:
            self.pool_up = torch.empty(N_EXPERTS, 2 * EXPERT_INNER, DIM, dtype=torch.uint8, device=device)
            self.pool_down = torch.empty(N_EXPERTS, DIM, EXPERT_INNER, dtype=torch.uint8, device=device)
        self._slot_of: "OrderedDict[Tuple[int, int], int]" = OrderedDict()  # LRU order
        self._free_slots = list(range(N_EXPERTS))
        self.bytes_copied = 0

    def add_layer(self, layer: int, up: torch.Tensor, down: torch.Tensor, up_scale, down_scale):
        """``up``/``down``: fp8 checkpoint tensors on the CPU."""
        self.up_scale[layer] = up_scale.to(self.device, torch.float32)
        self.down_scale[layer] = down_scale.to(self.device, torch.float32)
        up, down = up.view(torch.uint8), down.view(torch.uint8)
        if layer in self.offloaded:
            self.host_up[layer] = up.contiguous().pin_memory()
            self.host_down[layer] = down.contiguous().pin_memory()
        else:
            self.resident_up[layer] = up.to(self.device)
            self.resident_down[layer] = down.to(self.device)

    def fetch(self, layer: int, experts: List[int]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Return (up storage, down storage, index of each expert in that storage)."""
        if layer not in self.offloaded:
            return self.resident_up[layer], self.resident_down[layer], experts
        needed = set((layer, e) for e in experts)
        slots = []
        for e in experts:
            key = (layer, e)
            slot = self._slot_of.get(key)
            if slot is None:
                if self._free_slots:
                    slot = self._free_slots.pop()
                else:
                    # Evict the least recently used expert that this call does not need.
                    victim = next(k for k in self._slot_of if k not in needed)
                    slot = self._slot_of.pop(victim)
                self.pool_up[slot].copy_(self.host_up[layer][e], non_blocking=True)
                self.pool_down[slot].copy_(self.host_down[layer][e], non_blocking=True)
                self.bytes_copied += EXPERT_BYTES
                self._slot_of[key] = slot
            else:
                self._slot_of.move_to_end(key)
            slots.append(slot)
        return self.pool_up, self.pool_down, slots


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Moondream31:
    """Moondream 3.1 ``detect`` with RAM-offloaded experts."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        tokenizer=None,
        vram_reserve_gb: float = 0.9,
        max_resident_layers: Optional[int] = None,
        max_crops: int = MAX_CROPS,
        max_context: int = 1024,
        device: str = "cuda",
    ):
        from safetensors import safe_open

        if weights_path is None:
            from huggingface_hub import hf_hub_download

            weights_path = hf_hub_download(HF_REPO, "model.safetensors", revision=HF_REVISION)
        if tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_pretrained(TOKENIZER_REPO)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.max_crops = max_crops
        self.max_context = max_context

        with safe_open(weights_path, framework="pt") as f:
            def get(name: str) -> torch.Tensor:
                return f.get_tensor(name).to(self.device, self.dtype)

            self._load_dense(get)
            torch.cuda.synchronize(self.device)

            n_resident = self._resident_layers(vram_reserve_gb, max_resident_layers)
            self.experts = ExpertStore(n_resident, self.device)
            for layer in self.experts.moe_layers:
                p = f"text_model.transformer.h.{layer}"
                self.experts.add_layer(
                    layer,
                    f.get_tensor(f"{p}.mlp.experts.weight"),
                    f.get_tensor(f"{p}.mlp.output_experts.weight"),
                    f.get_tensor(f"{p}.moe_quant.up_scale"),
                    f.get_tensor(f"{p}.moe_quant.down_scale"),
                )
        torch.cuda.synchronize(self.device)

        offloaded = len(self.experts.offloaded)
        logger.info(
            "Moondream 3.1 loaded: %d/%d MoE layers in VRAM, %d in RAM (%.1f GB), %.2f GB VRAM allocated",
            n_resident,
            N_LAYERS - MOE_START,
            offloaded,
            offloaded * N_EXPERTS * EXPERT_BYTES / 1e9,
            torch.cuda.memory_allocated(self.device) / 1e9,
        )

        self._setup_caches()

    # -- loading -------------------------------------------------------------

    def _load_dense(self, get) -> None:
        v = "vision_encoder.encoder.model.visual"
        self.patch_emb = _Linear(get(f"{v}.patch_embed.linear.weight"), get(f"{v}.patch_embed.linear.bias"))
        self.pos_emb = get(f"{v}.pos_embed")
        self.vision_post_ln = _LayerNorm(get(f"{v}.norm.weight"), get(f"{v}.norm.bias"))
        self.vision_blocks = []
        for i in range(ENC_LAYERS):
            b = f"{v}.blocks.{i}"
            self.vision_blocks.append(
                {
                    "ln1": _LayerNorm(get(f"{b}.norm1.weight"), get(f"{b}.norm1.bias")),
                    "ln2": _LayerNorm(get(f"{b}.norm2.weight"), get(f"{b}.norm2.bias")),
                    "qkv": _Linear(get(f"{b}.attn.qkv.weight"), get(f"{b}.attn.qkv.bias")),
                    "proj": _Linear(get(f"{b}.attn.proj.weight"), get(f"{b}.attn.proj.bias")),
                    "fc1": _Linear(get(f"{b}.mlp.fc1.weight"), get(f"{b}.mlp.fc1.bias")),
                    "fc2": _Linear(get(f"{b}.mlp.fc2.weight"), get(f"{b}.mlp.fc2.bias")),
                }
            )
        p = "vision_encoder.projection.mlp"
        self.proj_fc1 = _Linear(get(f"{p}.fc1.weight"), get(f"{p}.fc1.bias"))
        self.proj_fc2 = _Linear(get(f"{p}.fc2.weight"), get(f"{p}.fc2.bias"))

        self.wte = get("text_model.transformer.embd.wte.weight")
        self.post_ln = _LayerNorm(get("text_model.lm_head.ln.weight"), get("text_model.lm_head.ln.bias"))
        self.lm_head = _Linear(get("text_model.lm_head.linear.weight"), get("text_model.lm_head.linear.bias"))
        self.blocks = []
        for i in range(N_LAYERS):
            b = f"text_model.transformer.h.{i}"
            block = {
                "ln": _LayerNorm(get(f"{b}.ln.weight"), get(f"{b}.ln.bias")),
                "qkv": _Linear(get(f"{b}.mixer.Wqkv.weight"), get(f"{b}.mixer.Wqkv.bias")),
                "proj": _Linear(get(f"{b}.mixer.out_proj.weight"), get(f"{b}.mixer.out_proj.bias")),
                "tau_wq": get(f"{b}.tau_wq"),
                "tau_wv": get(f"{b}.tau_wv"),
                "tau_alpha": get(f"{b}.tau_alpha").float(),
            }
            if i >= MOE_START:
                block["router"] = _Linear(get(f"{b}.gate.weight"), get(f"{b}.gate.bias"))
            else:
                block["fc1"] = _Linear(get(f"{b}.mlp.fc1.weight"), get(f"{b}.mlp.fc1.bias"))
                block["fc2"] = _Linear(get(f"{b}.mlp.fc2.weight"), get(f"{b}.mlp.fc2.bias"))
            self.blocks.append(block)

        r = "region_model"
        self.coord_encoder = _Linear(get(f"{r}.coordinate_encoder.weight"), get(f"{r}.coordinate_encoder.bias"))
        self.coord_decoder = _Linear(get(f"{r}.coordinate_head.weight"), get(f"{r}.coordinate_head.bias"))
        self.size_encoder = _Linear(get(f"{r}.size_encoder.weight"), get(f"{r}.size_encoder.bias"))
        self.size_decoder = _Linear(get(f"{r}.size_head.weight"), get(f"{r}.size_head.bias"))
        self.region_ln = _LayerNorm(get(f"{r}.ln.weight"), get(f"{r}.ln.bias"))
        self.coord_features = get(f"{r}.coordinate_features.weight").T.contiguous()  # [1, 128]
        self.size_features = get(f"{r}.size_features.weight").T.contiguous()  # [2, 256]

        freqs = 1.0 / (ROPE_THETA ** (torch.arange(0, ROT_DIM, 2, dtype=torch.float32) / ROT_DIM))
        angles = torch.arange(self.max_context, dtype=torch.float32)[:, None] * freqs[None]
        self.rope_cos = angles.cos().to(self.device)
        self.rope_sin = angles.sin().to(self.device)
        self.coord_eos_ids = torch.tensor([COORD_ID, EOS_ID], device=self.device)

    def _resident_layers(self, vram_reserve_gb: float, max_resident_layers: Optional[int]) -> int:
        n_moe = N_LAYERS - MOE_START
        free, _ = torch.cuda.mem_get_info(self.device)
        kv_bytes = 2 * N_LAYERS * N_HEADS * self.max_context * HEAD_DIM * 2
        budget = free - kv_bytes - ACTIVATION_HEADROOM_BYTES - int(vram_reserve_gb * 1e9)
        layer_bytes = N_EXPERTS * EXPERT_BYTES
        n = int(budget // layer_bytes)
        if n < n_moe:
            # Offloading needs a pool of one layer's worth of slots on the GPU.
            n = int((budget - layer_bytes) // layer_bytes)
        n = max(0, min(n_moe, n))
        if max_resident_layers is not None:
            n = min(n, int(max_resident_layers))
        if budget < (layer_bytes if n < n_moe else 0):
            raise MemoryError(
                f"Moondream needs ~{(layer_bytes + ACTIVATION_HEADROOM_BYTES) / 1e9:.1f} GB of free VRAM "
                f"beyond the dense weights, only {free / 1e9:.1f} GB free (reserve {vram_reserve_gb} GB)."
            )
        return n

    def _setup_caches(self) -> None:
        shape = (1, N_HEADS, self.max_context, HEAD_DIM)
        self.k_cache = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(N_LAYERS)]
        self.v_cache = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(N_LAYERS)]

    # -- text model ----------------------------------------------------------

    def _rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x_rot, x_pass = x[..., :ROT_DIM], x[..., ROT_DIM:]
        half = ROT_DIM // 2
        xr, xi = x_rot[..., :half], x_rot[..., half:]
        cos = self.rope_cos[pos][None, None]
        sin = self.rope_sin[pos][None, None]
        out = torch.stack((xr * cos - xi * sin, xr * sin + xi * cos), dim=-1).flatten(-2)
        return torch.cat([out.to(x.dtype), x_pass], dim=-1)

    def _attn(self, x: torch.Tensor, block: dict, layer: int, pos: torch.Tensor, mask) -> torch.Tensor:
        bsz, seq, _ = x.shape
        qkv = block["qkv"](x)
        q, k, v = qkv.split(DIM, dim=-1)
        q = q.view(bsz, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(bsz, seq, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(bsz, seq, N_HEADS, HEAD_DIM).transpose(1, 2)

        # Token- and position-dependent scaling of queries and values (tau attention).
        feat = _gelu_tanh(qkv)
        tok_q = torch.tanh(feat @ block["tau_wq"].T).permute(0, 2, 1)
        tok_v = torch.tanh(feat @ block["tau_wv"].T).permute(0, 2, 1)
        tau_pos = torch.sigmoid(block["tau_alpha"][:, None] * (pos.float() + 1).log()) + 0.5
        q = q * (tok_q + tau_pos[None].to(q.dtype)).unsqueeze(-1)
        v = v * (tok_v + tau_pos[None].to(v.dtype)).unsqueeze(-1)

        q, k = self._rope(q, pos), self._rope(k, pos)
        self.k_cache[layer][:, :, pos] = k
        self.v_cache[layer][:, :, pos] = v
        end = int(pos[-1]) + 1
        out = F.scaled_dot_product_attention(
            q, self.k_cache[layer][:, :, :end], self.v_cache[layer][:, :, :end], attn_mask=mask
        )
        return block["proj"](out.transpose(1, 2).reshape(bsz, seq, DIM))

    def _moe(self, x: torch.Tensor, block: dict, layer: int) -> torch.Tensor:
        bsz, seq, _ = x.shape
        x = x.reshape(-1, DIM)
        logits = block["router"](x)
        topk_logits, topk_idx = torch.topk(logits, TOP_K, dim=-1)
        topk_w = F.softmax(topk_logits, dim=-1, dtype=torch.float32).to(x.dtype)

        experts = sorted(set(topk_idx.view(-1).tolist()))
        up, down, slots = self.experts.fetch(layer, experts)
        up_scale, down_scale = self.experts.up_scale[layer], self.experts.down_scale[layer]

        if x.shape[0] == 1:
            slot_of = dict(zip(experts, slots))
            ids = topk_idx[0].tolist()
            index = torch.tensor([slot_of[e] for e in ids], device=self.device)
            w_up = up[index].view(torch.float8_e4m3fn).to(self.dtype)  # [K, 2I, D]
            w_down = down[index].view(torch.float8_e4m3fn).to(self.dtype)  # [K, D, I]
            h = torch.bmm(w_up, x.expand(TOP_K, -1).unsqueeze(-1)).squeeze(-1)
            h = h * up_scale[topk_idx[0]].to(self.dtype)
            gate, lin = h.chunk(2, dim=-1)
            h = F.gelu(gate) * (lin + 1)
            y = torch.bmm(w_down, h.unsqueeze(-1)).squeeze(-1) * down_scale[topk_idx[0]].to(self.dtype)
            return (y * topk_w[0][:, None]).sum(0).view(bsz, seq, DIM)

        out = torch.zeros_like(x)
        for expert, slot in zip(experts, slots):
            token_pos, which = (topk_idx == expert).nonzero(as_tuple=True)
            xt = x.index_select(0, token_pos)
            h = F.linear(xt, up[slot].view(torch.float8_e4m3fn).to(self.dtype))
            h = h * up_scale[expert].to(self.dtype)
            gate, lin = h.chunk(2, dim=-1)
            h = F.gelu(gate) * (lin + 1)
            y = F.linear(h, down[slot].view(torch.float8_e4m3fn).to(self.dtype))
            y = y * down_scale[expert].to(self.dtype)
            out.index_add_(0, token_pos, y * topk_w[token_pos, which][:, None])
        return out.view(bsz, seq, DIM)

    def _decoder(self, x: torch.Tensor, pos: torch.Tensor, mask=None) -> torch.Tensor:
        for layer, block in enumerate(self.blocks):
            h = block["ln"](x)
            a = self._attn(h, block, layer, pos, mask)
            if layer >= MOE_START:
                m = self._moe(h, block, layer)
            else:
                m = block["fc2"](_gelu_tanh(block["fc1"](h)))
            x = x + a + m
        return x

    # -- vision --------------------------------------------------------------

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        crops, tiling = _overlap_crops(np.asarray(image.convert("RGB")), self.max_crops)
        x = torch.from_numpy(crops).to(self.device).permute(0, 3, 1, 2).to(self.dtype)
        x = x.div_(255.0).sub_(0.5).div_(0.5)

        n = x.shape[0]
        x = x.reshape(n, 3, GRID, PATCH, GRID, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(n, GRID * GRID, -1)
        x = self.patch_emb(x) + self.pos_emb
        for blk in self.vision_blocks:
            h = blk["ln1"](x)
            q, k, v = (
                t.view(n, -1, ENC_HEADS, ENC_DIM // ENC_HEADS).transpose(1, 2) for t in blk["qkv"](h).chunk(3, -1)
            )
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, -1, ENC_DIM)
            x = x + blk["proj"](a)
            x = x + blk["fc2"](_gelu_tanh(blk["fc1"](blk["ln2"](x))))
        x = self.vision_post_ln(x)

        global_features = x[0]
        local = _reconstruct(x[1:].view(-1, GRID, GRID, ENC_DIM), tiling)
        local = F.adaptive_avg_pool2d(local.permute(2, 0, 1), (GRID, GRID)).permute(1, 2, 0)
        features = torch.cat([global_features, local.reshape(GRID * GRID, ENC_DIM)], dim=-1)
        return self.proj_fc2(_gelu_tanh(self.proj_fc1(features)))

    # -- region head ---------------------------------------------------------

    @staticmethod
    def _fourier(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        f = 2 * math.pi * x @ w
        return torch.cat([f.cos(), f.sin()], dim=-1)

    def _embed_coord(self, value: torch.Tensor) -> torch.Tensor:
        return self.coord_encoder(self._fourier(value.view(1, 1).to(self.dtype), self.coord_features)).view(1, 1, DIM)

    def _embed_size(self, wh: torch.Tensor) -> torch.Tensor:
        return self.size_encoder(self._fourier(wh.view(1, 2).to(self.dtype), self.size_features)).view(1, 1, DIM)

    # -- public API ----------------------------------------------------------

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> None:
        """Prefill BOS + image into the KV cache. Later ``detect`` calls reuse it."""
        img = self._encode_image(image)
        bos = self.wte[BOS_ID].view(1, 1, DIM)
        x = torch.cat([bos, img[None]], dim=1)
        pos = torch.arange(IMAGE_PREFIX, device=self.device)
        self._decoder(x, pos)  # the image prefix attends bidirectionally: no mask

    @torch.inference_mode()
    def detect(self, label: str, max_objects: int = 50) -> List[dict]:
        """Boxes (normalized x_min, y_min, x_max, y_max) of ``label`` in the encoded image."""
        ids = DETECT_PREFIX + self.tokenizer.encode(" " + label).ids + DETECT_SUFFIX
        tokens = torch.tensor(ids, device=self.device)
        n = len(ids)
        pos = torch.arange(IMAGE_PREFIX, IMAGE_PREFIX + n, device=self.device)
        mask = torch.ones(n, IMAGE_PREFIX + n, dtype=torch.bool, device=self.device)
        mask[:, IMAGE_PREFIX:] = torch.tril(mask[:, IMAGE_PREFIX:])
        hidden = self._decoder(F.embedding(tokens, self.wte)[None], pos, mask[None, None])[:, -1:]
        cur = IMAGE_PREFIX + n
        limit = min(max_objects, (self.max_context - cur - 1) // 3)

        objects = []
        is_coord = self._next_is_coord(hidden)
        while is_coord and len(objects) < limit:
            x_c = self._decode_coord(hidden)
            hidden = self._step(self._embed_coord(x_c), cur)
            cur += 1
            y_c = self._decode_coord(hidden)
            hidden = self._step(self._embed_coord(y_c), cur)
            cur += 1
            size_logits = self.size_decoder(self.region_ln(hidden)).view(2, -1)
            wh = torch.pow(2.0, size_logits.argmax(-1).float() / 1023.0 * 10.0 - 10.0)
            hidden = self._step(self._embed_size(wh), cur)
            cur += 1

            xc, yc, (w, h) = float(x_c), float(y_c), wh.tolist()
            objects.append({"x_min": xc - w / 2, "y_min": yc - h / 2, "x_max": xc + w / 2, "y_max": yc + h / 2})
            is_coord = self._next_is_coord(hidden)
        return objects

    def _step(self, emb: torch.Tensor, position: int) -> torch.Tensor:
        return self._decoder(emb, torch.tensor([position], device=self.device))

    def _decode_coord(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.coord_decoder(self.region_ln(hidden)).view(-1)
        return logits.argmax() / logits.numel()

    def _next_is_coord(self, hidden: torch.Tensor) -> bool:
        """Next token restricted to {coord, eos}: another object, or the end."""
        h = self.post_ln(hidden[0, -1])
        logits = h @ self.lm_head.weight[self.coord_eos_ids].T + self.lm_head.bias[self.coord_eos_ids]
        return bool(logits.argmax() == 0)
