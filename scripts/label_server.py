#!/usr/bin/env python3
"""
Web UI for robot color labeling (Flask + browser).

  python scripts/label_server.py -i training/robot_pics -c red blue yellow white unknown

Open http://127.0.0.1:5055 — same workflow as scripts/label.py (t / l / s / n / z).

When everything is already labeled, browse saved labels with:

  python scripts/label_server.py -i ... -c ... --review-uncertain   # hint ≠ saved label (needs .pth)
  python scripts/label_server.py -i ... -c ... --review-all           # all labeled, file order

Options: --hints, --uncertain-first, --shuffle, --auto-retrain-every, -o.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]


def _load_label_module():
    spec = importlib.util.spec_from_file_location("mario_label", ROOT / "scripts" / "label.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab = _load_label_module()


class LabelWorkspace:
    """Mutable session state (single-user local server)."""

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        colors: List[str],
        *,
        shuffle: bool,
        uncertain_first: bool,
        load_hints: bool,
        auto_retrain_every: int,
        review_uncertain: bool = False,
        review_all: bool = False,
        review_top: int = 500,
    ):
        import torch

        self.input_dir = input_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.colors = colors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.auto_retrain_every = max(0, auto_retrain_every)

        self.images_all: List[Path] = sorted(self.input_dir.glob("*.png")) + sorted(
            self.input_dir.glob("*.jpg")
        )
        self.labels: Dict[str, str] = lab.load_labels(self.output_dir)
        self.browse_labeled = False
        self.review_mode: Optional[str] = None
        self.review_banner = ""

        if review_uncertain:
            self.model_state = lab.load_or_none(colors, self.device)
            if self.model_state is None:
                raise ValueError(
                    "Serve data/models/color_classifier_active.pth per --review-uncertain "
                    "(addestra prima con train_color o label t)."
                )
            labeled_paths = [p for p in self.images_all if p.name in self.labels]
            if not labeled_paths:
                raise ValueError("Nessuna etichetta in labels.json: nulla da rivedere.")
            print(f"[review] hint vs etichetta su {len(labeled_paths)} immagini…", flush=True)
            model = self.model_state["model"]
            i2c = self.model_state["idx2color"]
            try:
                idxs = lab.predict_class_indices_batch(
                    labeled_paths, model, self.device, show_progress=True
                )
                confs = lab.predict_max_confidence_batch(
                    labeled_paths, model, self.device, show_progress=False
                )
            except Exception:
                idxs, confs = [], []
                for p in labeled_paths:
                    try:
                        ix, cf = lab.predict(model, p, self.device)
                        idxs.append(ix)
                        confs.append(cf)
                    except Exception:
                        idxs.append(-1)
                        confs.append(0.0)
            mismatches = []
            for p, ix, cf in zip(labeled_paths, idxs, confs):
                if ix < 0:
                    continue
                hint = lab.resolve_hint_color(ix, float(cf), i2c, self.colors)
                if lab.label_disagrees(hint, self.labels[p.name]):
                    mismatches.append((cf, p))
            mismatches.sort(key=lambda x: x[0])
            self.queue = [p for _, p in mismatches[: max(1, review_top)]]
            if not self.queue:
                raise ValueError(
                    "Nessun mismatch tra hint e labels.json: nulla da rivedere con --review-uncertain."
                )
            self.browse_labeled = True
            self.review_mode = "uncertain"
            self.review_banner = (
                f"Revisione mismatch: {len(self.queue)} immagini (hint ≠ etichetta, conf modello ↑)"
            )
        elif review_all:
            self.queue = [p for p in self.images_all if p.name in self.labels]
            self.browse_labeled = True
            self.review_mode = "all"
            self.review_banner = f"Modalità revisione: tutte le etichette ({len(self.queue)} immagini)"
            need_ckpt = load_hints
            self.model_state = lab.load_or_none(colors, self.device) if need_ckpt else None
        else:
            need_ckpt = load_hints or uncertain_first
            self.model_state = lab.load_or_none(colors, self.device) if need_ckpt else None

            self.queue = lab.build_unlabeled_queue(
                self.images_all,
                self.labels,
                self.model_state,
                self.device,
                shuffle=shuffle,
                uncertain_first=uncertain_first,
            )

        self.i = 0
        self.test_acc: Optional[float] = None
        self.lock = threading.Lock()

    def images_dir(self) -> Path:
        return self.input_dir

    def _sync_index(self) -> None:
        if self.browse_labeled:
            return
        while self.i < len(self.queue) and self.queue[self.i].name in self.labels:
            self.i += 1

    def _current_path(self) -> Optional[Path]:
        self._sync_index()
        if self.i >= len(self.queue):
            return None
        return self.queue[self.i]

    def _hint(self) -> Optional[Dict[str, Any]]:
        p = self._current_path()
        if p is None or self.model_state is None:
            return None
        try:
            idx_pred, conf = lab.predict(self.model_state["model"], p, self.device)
            cname = lab.resolve_hint_color(
                idx_pred, conf, self.model_state["idx2color"], self.colors
            )
            return {"color": cname, "conf": float(conf)}
        except Exception:
            return None

    def state(self) -> Dict[str, Any]:
        with self.lock:
            self._sync_index()
            n_lab = len(self.labels)
            total = len(self.queue)
            if self.i >= total:
                return {
                    "done": True,
                    "n_labeled": n_lab,
                    "total": total,
                    "colors": self.colors,
                    "test_acc": self.test_acc,
                    "review_mode": self.review_mode,
                    "review_banner": self.review_banner,
                }
            p = self.queue[self.i]
            saved = self.labels.get(p.name)
            h = self._hint()
            hint_mismatch = bool(
                h
                and saved
                and lab.label_disagrees(h["color"], saved)
            )
            return {
                "done": False,
                "filename": p.name,
                "image_url": f"/crop/{p.name}",
                "index": self.i,
                "total": total,
                "n_labeled": n_lab,
                "colors": self.colors,
                "hint": h,
                "test_acc": self.test_acc,
                "review_mode": self.review_mode,
                "review_banner": self.review_banner,
                "saved_label": saved if self.browse_labeled else None,
                "hint_label_mismatch": hint_mismatch,
            }

    def label(self, color: str) -> str:
        if color not in self.colors:
            raise ValueError("colore non valido")
        with self.lock:
            p = self._current_path()
            if p is None:
                return "fine coda"
            self.labels[p.name] = color
            lab.save_labels(self.output_dir, self.labels)
            lab.mark_human(self.output_dir, self.labels, p.name)
            self.model_state, new_acc = lab.maybe_auto_retrain(
                self.labels,
                self.images_dir(),
                self.colors,
                self.model_state,
                self.device,
                self.auto_retrain_every,
            )
            if new_acc is not None:
                self.test_acc = float(new_acc)
            self.i += 1
        return ""

    def confirm_hint(self) -> str:
        with self.lock:
            h = self._hint()
            p = self._current_path()
            if not h or p is None:
                raise RuntimeError("nessun hint")
            self.labels[p.name] = h["color"]
            lab.save_labels(self.output_dir, self.labels)
            lab.mark_human(self.output_dir, self.labels, p.name)
            self.model_state, new_acc = lab.maybe_auto_retrain(
                self.labels,
                self.images_dir(),
                self.colors,
                self.model_state,
                self.device,
                self.auto_retrain_every,
            )
            if new_acc is not None:
                self.test_acc = float(new_acc)
            self.i += 1
        return ""

    def skip(self) -> None:
        with self.lock:
            if self._current_path() is None:
                return
            self.i += 1

    def back(self) -> None:
        with self.lock:
            self.i = max(0, self.i - 1)

    def train(self) -> str:
        with self.lock:
            ms, acc = lab.do_retrain(
                self.labels,
                self.images_dir(),
                self.colors,
                self.model_state,
                self.device,
                merged=False,
            )
            self.model_state = ms
            if acc is not None:
                self.test_acc = float(acc)
        return (
            f"Train completato. Test ≈ {self.test_acc * 100:.1f}%"
            if self.test_acc is not None
            else "Train completato."
        )

    def autolabel_train(self) -> str:
        with self.lock:
            if self.model_state is None:
                raise RuntimeError("Serve un modello: usa Train (t) prima.")
            lab._autolabel(
                self.queue,
                self.labels,
                self.model_state,
                self.output_dir,
                self.colors,
                self.device,
                replace_all=True,
                images_dir=self.input_dir,
            )
            ms, acc = lab.do_retrain(
                self.labels,
                self.images_dir(),
                self.colors,
                self.model_state,
                self.device,
                merged=True,
            )
            self.model_state = ms
            if acc is not None:
                self.test_acc = float(acc)
        return "Rilabel su tutta la cartella e training completati."

    def sort_rest(self) -> str:
        with self.lock:
            remaining = self.queue[self.i :]
            if not remaining:
                return "Niente da ordinare."
            if self.model_state is None:
                raise RuntimeError("Serve un modello (Train prima).")
            try:
                confs = lab.predict_max_confidence_batch(
                    remaining, self.model_state["model"], self.device, show_progress=False
                )
                scored = list(zip(confs, remaining))
            except Exception:
                scored = []
                for p in remaining:
                    try:
                        _, conf = lab.predict(self.model_state["model"], p, self.device)
                        scored.append((conf, p))
                    except Exception:
                        scored.append((0.0, p))
            scored.sort(key=lambda x: x[0])
            self.queue[:] = self.queue[: self.i] + [p for _, p in scored]
        return f"Ordinate {len(remaining)} immagini per confidenza."


def create_app(ws: LabelWorkspace) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
    )

    @app.route("/")
    def index():
        return render_template("label_web.html")

    @app.route("/api/state")
    def api_state():
        try:
            return jsonify(ws.state())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/crop/<path:name>")
    def crop(name):
        safe = Path(name).name
        base = ws.input_dir.resolve()
        path = (base / safe).resolve()
        if not str(path).startswith(str(base)):
            abort(403)
        if not path.is_file():
            abort(404)
        return send_from_directory(ws.input_dir, safe)

    def _json():
        return request.get_json(force=True, silent=True) or {}

    @app.route("/api/label", methods=["POST"])
    def api_label():
        try:
            c = _json().get("color")
            if not c:
                return jsonify({"error": "missing color"}), 400
            msg = ws.label(c)
            return jsonify({"ok": True, "message": msg})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/confirm_hint", methods=["POST"])
    def api_confirm():
        try:
            ws.confirm_hint()
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/skip", methods=["POST"])
    def api_skip():
        ws.skip()
        return jsonify({"ok": True})

    @app.route("/api/back", methods=["POST"])
    def api_back():
        ws.back()
        return jsonify({"ok": True})

    @app.route("/api/train", methods=["POST"])
    def api_train():
        try:
            msg = ws.train()
            return jsonify({"ok": True, "message": msg})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/autolabel_train", methods=["POST"])
    def api_autolabel():
        try:
            msg = ws.autolabel_train()
            return jsonify({"ok": True, "message": msg})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/sort", methods=["POST"])
    def api_sort():
        try:
            msg = ws.sort_rest()
            return jsonify({"ok": True, "message": msg})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    return app


def main() -> int:
    p = argparse.ArgumentParser(description="Label robot crops in the browser (Flask).")
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-c", "--colors", nargs="+", required=True)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5055)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--hints", action="store_true")
    p.add_argument("--uncertain-first", action="store_true")
    p.add_argument(
        "--auto-retrain-every",
        type=int,
        default=lab.DEFAULT_AUTO_RETRAIN_EVERY,
    )
    p.add_argument(
        "--review-uncertain",
        action="store_true",
        help="Solo mismatch hint vs etichetta salvata (tipico errore autolabel), conf ↑ (serve .pth).",
    )
    p.add_argument(
        "--review-all",
        action="store_true",
        help="Rivedi tutte le immagini che hanno un'etichetta in labels.json (ordine file).",
    )
    p.add_argument(
        "--review-top",
        type=int,
        default=500,
        help="Con --review-uncertain, massimo quante immagini in coda (default 500).",
    )
    args = p.parse_args()

    if not args.input.is_dir():
        print(f"Not a directory: {args.input}", file=sys.stderr)
        return 1

    if args.review_uncertain and args.review_all:
        print("Usa solo uno tra --review-uncertain e --review-all.", file=sys.stderr)
        return 1

    out = args.output or args.input
    try:
        ws = LabelWorkspace(
            args.input,
            out,
            args.colors,
            shuffle=args.shuffle,
            uncertain_first=args.uncertain_first,
            load_hints=args.hints,
            auto_retrain_every=max(0, args.auto_retrain_every),
            review_uncertain=args.review_uncertain,
            review_all=args.review_all,
            review_top=max(1, args.review_top),
        )
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    app = create_app(ws)
    print(f"Apri il browser: http://{args.host}:{args.port}/")
    print(f"Immagini: {len(ws.images_all)}  Etichette in JSON: {len(ws.labels)}  In coda: {len(ws.queue)}")
    if ws.review_banner:
        print(ws.review_banner)
    elif not ws.browse_labeled and not ws.queue and ws.labels:
        print(
            "Suggerimento: tutte le immagini risultano già etichettate. "
            "Rilancia con --review-uncertain o --review-all per verificare.",
            file=sys.stderr,
        )
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
