import argparse
import cv2
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.streaming import parse_deltas

BASE_DIR = Path(__file__).parents[1] / "data" / "games"

# --- Helper Classes and Functions ---

class AddVideoAction(argparse.Action):
    """
    A custom argparse action that builds a single list made up of both -v and -y, preserving the order.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # Create the queue list if it doesn't exist yet
        if getattr(namespace, "queue", None) is None:
            setattr(namespace, "queue", [])

        if option_string in ("-v", "--video"):
            if values.startswith("http://") or values.startswith("https://"):
                namespace.queue.append({"type": "direct", "url": values})
            else:
                namespace.queue.append({"type": "local", "url": values})
        elif option_string in ("-y", "--youtube"):
            # values[0] is the URL, values[1] is the time range string
            namespace.queue.append({
                "type": "youtube",
                "url": values[0],
                "time_range": values[1]
            })

class ErrorOnlyLogger:
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): print(msg, file=sys.stderr)

class FFmpegProgressThread(threading.Thread):
    def __init__(self, progress_path, duration, desc="Downloading"):
        super().__init__()
        self.progress_path = progress_path
        self.duration = duration
        self.desc = desc
        self.stop_event = threading.Event()

    def run(self):
        pbar = tqdm(total=self.duration, unit='s', desc=self.desc)
        
        while not self.progress_path.exists() and not self.stop_event.is_set():
            time.sleep(0.1)
            
        if not self.progress_path.exists():
            pbar.close()
            return

        with self.progress_path.open('r') as f:
            last_time = 0
            while not self.stop_event.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                
                if line.startswith("out_time_us="):
                    try:
                        current_time = int(line.split("=")[1]) / 1_000_000
                        
                        if current_time < last_time - 2:
                            pbar.n = 0
                            pbar.desc = f"{self.desc} (Audio/Muxing)"
                        
                        last_time = current_time
                        pbar.n = min(current_time, self.duration)
                        pbar.refresh()
                    except ValueError:
                        pass
                        
        pbar.close()

    def stop(self):
        self.stop_event.set()


# --- Main Script ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "game_name",
        help="The name of the game (creates the main folder)"
    )
    parser.add_argument(
        "-v", "--video",
        action=AddVideoAction,
        metavar="URL",
        help="Direct video URL"
    )
    parser.add_argument(
        "-y", "--youtube",
        action=AddVideoAction,
        nargs=2,
        metavar=("URL", "TIME_RANGE"),
        help="YouTube URL and time range (e.g. -y https://youtu.be/... 00:10-01:30)"
    )
    parser.add_argument(
        "-g", "--gamecontroller",
        nargs="+",
        help="URL to a YAML file, path to a local YAML file, OR path to a ZIP file followed by the internal file path"
    )

    args = parser.parse_args()

    # validation checks
    if not getattr(args, "queue", None):
        raise ValueError("No video provided")

    if len(args.queue) > 26:
        raise ValueError("Too many videos, only up to 26")

    game_dir = BASE_DIR / args.game_name
    if game_dir.exists():
        resp = input(f"Game {game_dir} already exists. DELETE IT ENTIRELY? Type yes to confirm / no to abort > ")
        if resp.lower() == "y" or resp.lower() == "yes":
            shutil.rmtree(game_dir)
        else:
            raise FileExistsError("Trollollollo")

    game_dir.mkdir()

    # Handle the GameController log
    if args.gamecontroller:
        primary_arg = args.gamecontroller[0]
        gc_target_path = game_dir / "gc.yaml"

        if len(args.gamecontroller) == 1:
            if primary_arg.startswith("http://") or primary_arg.startswith("https://"):
                # It's a remote URL download
                urllib.request.urlretrieve(primary_arg, gc_target_path)
            else:
                # It's a direct YAML file copy
                shutil.copy(primary_arg, gc_target_path)
        elif len(args.gamecontroller) == 2:
            try:
                # It's a ZIP extraction
                zip_path, internal_path = args.gamecontroller
                with zipfile.ZipFile(zip_path, 'r') as z:
                    if internal_path not in z.namelist():
                        print(f"🚨 Error: '{internal_path}' not found inside '{zip_path}'.", file=sys.stderr)
                        sys.exit(1)

                    with z.open(internal_path) as source, open(gc_target_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
            except zipfile.BadZipFile as e:
                raise zipfile.BadZipFile("-g with two arguments expects a zip file as the first but this was not") from e
        else:
            parser.error("The -g flag takes a maximum of 2 arguments.")


    for index, item in enumerate(args.queue):
        section_letter = chr(65 + index)
        section_name = f"mario_{section_letter}"
        section_dir = game_dir / section_name
        section_dir.mkdir()
        video_path = section_dir / "video.mp4"

        try:
            if item["type"] == "local":
                source_path = item["url"]                
                if source_path.lower().endswith(".mp4"):
                    shutil.copy(source_path, video_path)
                    print(f"Video {index + 1}/{len(args.queue)}: copied")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".log") as progress_file:
                        progress_path = Path(progress_file.name)
                        command = [
                            "ffmpeg", 
                            "-y",
                            "-progress", str(progress_path),
                            "-i", source_path,
                            video_path
                        ]
                        x = cv2.VideoCapture(str(source_path))
                        tracker = FFmpegProgressThread(
                            progress_path=progress_path, 
                            duration=x.get(cv2.CAP_PROP_FRAME_COUNT) / x.get(cv2.CAP_PROP_FPS), 
                            desc=f"Video {index + 1}/{len(args.queue)}"
                        )
                        tracker.start()
                        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        tracker.stop()
                        tracker.join()
                        
            elif item["type"] == "direct":
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=f"Video {index + 1}/{len(args.queue)}") as progress_bar:
                    def reporthook(block_num, block_size, total_size):
                        if progress_bar.total is None and total_size > 0:
                            progress_bar.total = total_size
                        downloaded = block_num * block_size
                        progress_bar.update(downloaded - progress_bar.n)
                    urllib.request.urlretrieve(item["url"], video_path, reporthook=reporthook)

            elif item["type"] == "youtube":
                start, end = parse_deltas(item["time_range"])
                start_sec = start.total_seconds()
                end_sec = end.total_seconds()
                duration = end_sec - start_sec

                with tempfile.NamedTemporaryFile(suffix=".log") as progress_file:
                    progress_path = Path(progress_file.name)
                    ydl_opts = {
                        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                        'outtmpl': str(video_path),
                        'download_ranges': download_range_func(None, [(start_sec, end_sec)]),
                        'force_keyframes_at_cuts': True,
                        'logger': ErrorOnlyLogger(),
                        'external_downloader_args': {'ffmpeg': ['-loglevel', 'error', '-progress', str(progress_path)]},
                        'postprocessor_args': {'ffmpeg': ['-loglevel', 'error', '-progress', str(progress_path)]},
                    }

                    tracker = FFmpegProgressThread(
                        progress_path=progress_path, 
                        duration=duration, 
                        desc=f"Video {index + 1}/{len(args.queue)}"
                    )
                    tracker.start()

                    try:
                        with YoutubeDL(ydl_opts) as ydl:
                            ydl.download([item["url"]])
                    except Exception as e:
                        print(f"\n❌ Failed to download {item['url']}. Error: {e}")
                    finally:
                        tracker.stop()
                        tracker.join()

        except Exception as e:
            print(f"❌ Failed to download {item['url']}. Error: {e}\n", file=sys.stderr)

if __name__ == "__main__":
    main()
