# MARIO - Multi-Agent RoboCup Interactive Observation

**Version 2.0.0** - Complete refactored and modernized codebase

MARIO is a computer vision system for analyzing RoboCup humanoid robot soccer matches. It processes video footage to detect, track, and classify robots and balls, then projects their positions onto a 2D field plan view for analysis.

## Installation

### With pixi (recommended)

Install [pixi](https://pixi.sh) if you don't have it:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Then:

```bash
git clone <repository-url>
cd mario
pixi install
```

Run scripts inside the pixi environment:

```bash
pixi run python scripts/main.py <command line arguments>
```

### With conda (legacy)

```bash
git clone <repository-url>
cd mario
conda env create -f environment.yml [TODO: check depencies]
conda activate mario
```

Install ollama with snap (or following the guide on their website)

```bash
ollama pull glm-ocr:latest
ollama pull gemma3:1b
```

## Quick Start

### 1. Configure Your Project

Edit `config.yaml` to point to your video and data:

```yaml
video:
  game_name: "bhuman-htwk-BERLINVIDEO"
  video_fname: "video.mp4"

calibration:
  homography_timestamp_ms: 60000  # Timestamp for calibration frame
```

### 2. Prepare Your Data

Organize your data directory:

```
data/
├── models/
│   ├── best_yolov8.pt       # YOLOv8 weights for ball detection
│   ├── best_yolov12.pt      # YOLOv12 weights for robot detection
│   └── cnn_colori_best.pth  # Color classifier weights
├── config/
│   ├── bytetrack.yaml       # ByteTrack configuration
│   └── cnn_colori_config.yaml
├── assets/
│   └── field_new.png        # 2D field template image
└── games/
    └── your-game-name/
        └── video.mp4
```

