#!/usr/bin/env python3

import csv
import os
import subprocess

OUTPUT_DIR = "./downloads"

with open("dataset.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        competition = row["competition"]
        date = row["date"]
        match = row["match"]
        video_url = row["video"]
        log_url = row["log"]

        match_dir = os.path.join(OUTPUT_DIR, competition, date, match)
        os.makedirs(match_dir, exist_ok=True)

        video_filename = os.path.basename(video_url)
        log_filename = os.path.basename(log_url)

        print(f"Downloading: {competition}/{date}/{match}")
        subprocess.run(["wget", "-c", "-O", f"{match_dir}/{video_filename}", video_url])
        subprocess.run(["wget", "-c", "-O", f"{match_dir}/{log_filename}", log_url])

print("Done!")
