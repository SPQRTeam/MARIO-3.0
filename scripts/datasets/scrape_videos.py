#!/usr/bin/env python3
"""Scrape video listings from directory index."""

import re
from datetime import datetime, timedelta
from urllib.parse import urljoin
from utils import TEAM_ID_TO_NAME
import requests
import csv

def remote_file_size(url: str, timeout: float = 15.0) -> int | None:
    ##GENERATED
    r = requests.head(url, allow_redirects=True, timeout=timeout)
    r.raise_for_status()

    cl = r.headers.get("Content-Length")
    if cl is not None:
        return int(cl)
    r = requests.get(url, headers={"Range": "bytes=0-0"}, stream=True,
                     allow_redirects=True, timeout=timeout)
    r.raise_for_status()

    cr = r.headers.get("Content-Range")
    if cr and "/" in cr:
        return int(cr.split("/")[-1])

    cl = r.headers.get("Content-Length")
    return int(cl) if cl is not None else None



def scrape_logs(url: str) -> dict[str, list[str]]:
    html = requests.get(url).text
    # Find subdirectories (./FieldA/, ./FieldB/, etc.) (CHECK HTML)
    field_dirs = re.findall(r'href="\./?(Field[A-Z]/)?"', html)
    logs = []
    for field in field_dirs:
        field_url = urljoin(url, field)
        field_html = requests.get(field_url).text

        # Find all .yaml files
        logfiles = re.findall(r'href="([^"]+\.yaml)"', field_html)
        for f in logfiles:
            logs.append(str(urljoin(field_url, f)))

    return logs



def scrape_videos(url: str) -> dict[str, list[str]]:
    html = requests.get(url).text
    # Find subdirectories (FieldA/, FieldB/, etc.)
    field_dirs = re.findall(r'href="(Field[A-Z]/)?"', html)
    videos = {}
    for field in field_dirs:
        field_url = urljoin(url, field)
        field_html = requests.get(field_url).text

        mp4_files = re.findall(r'href="([^"]+\.mp4)"', field_html)
        field_name = field.rstrip("/")
        videos[field_name] = [urljoin(field_url, f) for f in mp4_files]

    return videos

def find_id_teams(id_match: str) -> list[str]:
    id_match = id_match.replace(".mp4", "")
    video_name = id_match.split("/")[-1]
    home_team = video_name.split("_")[-3]
    away_team = video_name.split("_")[-1]
    return [home_team, away_team]

def extract_datetime_from_video(url: str) -> datetime:
    # pi111_2025-07-17_14_21_21_19_vs_54.mp4
    name = url.split("/")[-1].replace(".mp4", "")
    parts = name.split("_")
    date_str = parts[1]  # 2025-07-17
    time_str = f"{parts[2]}:{parts[3]}:{parts[4]}"  # 14:21:21
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

def extract_datetime_from_log(url: str) -> datetime:
    # log_2025-07-17_10-21-17_SPQR-Team_WisTex-United.yaml
    name = url.split("/")[-1].replace(".yaml", "")
    parts = name.split("_")
    date_str = parts[1]  # 2025-07-17
    time_str = parts[2].replace("-", ":")  # 10:21:17
    return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

def find_matches(key_teams: frozenset, matches: dict, logs: dict) -> list[tuple[str, str]]:
    key_list = sorted(key_teams)
    print(key_list)
    found_videos = []
    for match in matches:
        if set(find_id_teams(match)) == set(key_teams):
            size = remote_file_size(match)
            #100MB
            if size > 100000000:
                found_videos.append(match)

    name1 = TEAM_ID_TO_NAME[int(key_list[0])] + "_" + TEAM_ID_TO_NAME[int(key_list[1])]
    name2 = TEAM_ID_TO_NAME[int(key_list[1])] + "_" + TEAM_ID_TO_NAME[int(key_list[0])]
    competition = match.split("/")[-4].split("_")[-1]
    print(competition)
    candidate_logs = [log for log in logs if name1 in log or name2 in log]

    # Match video al log più vicino (con offset di 4 ore per fuso orario)
    print("Found videos", found_videos)
    matched_pairs = []
    for video in found_videos:
        video_dt = extract_datetime_from_video(video) - timedelta(hours=4)
        video_date = video_dt.date()
        best_log = min(candidate_logs, key=lambda l: abs(extract_datetime_from_log(l) - video_dt), default=None)
        if best_log and abs(extract_datetime_from_log(best_log) - video_dt) < timedelta(minutes=60):
            matched_pairs.append({'competition': competition, 'date': str(video_date), 'match': name1 ,'video': video, 'log': best_log})
            print(f"MATCH: {video.split('/')[-1]} <-> {best_log.split('/')[-1]}")

    print("*"*60)
    print(matched_pairs)
    return matched_pairs

def mp_sort_key(mp):
    return mp["competition"] + mp["date"] + mp["match"]

if __name__ == "__main__":
    url_videos = "https://logs.berlin-united.com/2025-07-15_RC25/videos/"
    url_logs = "https://b-human.informatik.uni-bremen.de/public/logs/2025/RoboCup/gc/"
    videos = scrape_videos(url_videos)
    logs = scrape_logs(url_logs)
    all_videos = [vid for vids in videos.values() for vid in vids]
    keys_to_find = set(frozenset(find_id_teams(vid)) for vid in all_videos if "70" not in find_id_teams(vid))  # B-Team (70) is a dummy team
    matched_pairs = []
    for key in keys_to_find:
        matched_pairs.extend(find_matches(key, all_videos, logs))

    matched_pairs.sort(key=mp_sort_key)
    
    with open('dataset.csv', 'w') as csvfile:
        fieldnames = ['competition', 'date', 'match', 'video', 'log']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_pairs)
