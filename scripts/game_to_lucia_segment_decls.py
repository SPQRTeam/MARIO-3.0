# a helper script to connect with a different project whose current codename is "lucia-wm"

#!/usr/bin/env python3
import argparse
from pathlib import Path

ROOTDIR = (Path(__file__).parents[1] / "data" / "games").resolve()
PATHPREFIX = ROOTDIR
PATHSUFFIX = "orientazioniReduxOutput/merged.csv"

def main():
    parser = argparse.ArgumentParser(description="mario2lucia generator")
    parser.add_argument('-g', '--games', nargs='+', required=True, help="List of game names")
    args = parser.parse_args()

    for game_name in args.games:
        game_path = ROOTDIR / game_name
            
        for item in game_path.iterdir():
            if item.is_dir() and item.name.startswith("gc_section_"):
                print(f"{game_name}/{item.name}: {PATHPREFIX}/{game_name}/{item.name}/{PATHSUFFIX}")

if __name__ == "__main__":
    main()