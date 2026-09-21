from pathlib import Path
import shutil

def _size(f: Path):
    if f.is_file():
        return f.stat().st_size
    else:
        return sum(_size(child) for child in f.iterdir())

SUFFIXES = " kMGTP"
def _formatsize(x):
    x = float(x)
    si = 0
    while x >= 1024 and si < len(SUFFIXES) - 1:
        x /= 1024
        si += 1
    return f"{x:.2f} {SUFFIXES[si]}b"

GAMES_DIR = Path(__file__).parents[1] / "data" / "games"

size_freed = 0

for game_dir in GAMES_DIR.iterdir():
    for wipdir in game_dir.glob("WIPario_*"):
        size_freed += _size(wipdir)
        shutil.rmtree(wipdir)
        print(wipdir)

print(_formatsize(size_freed), "deleted! Yay!")
