import csv
from dataclasses import dataclass
import shutil
import traceback

import cv2
from pathlib import Path

@dataclass
class MultiWriterCSVParams:
    path: any
    header: any
@dataclass
class MultiWriterVideoParams:
    path: any
    fourcc: any
    fps: any
    width: any
    height: any

class MultiWriter:
    # Give only the paths you want to write to, leave the rest as None
    def __init__(self, csv_params=None, video_params=None, plan_view_params=None):
        self.csv_params = csv_params

        self.video_params = video_params
        self.plan_view_params = plan_view_params

        self.open_files = []
        self.csv_file = None
        self.csv_writer = None

        self.open_videos = []
        self.video_writer = None
        self.plan_view_writer = None

    @staticmethod
    def _open_video(params):
        w = cv2.VideoWriter(
            params.path,
            params.fourcc,
            params.fps,
            (params.width, params.height),
        )
        if not w.isOpened():
            raise IOError(f"Failed to open video file: {params.path}")
        return w

    def __enter__(self):
        try:
            if self.csv_params:
                self.csv_file = open(self.csv_params.path, "w", newline="")
                self.csv_writer = csv.writer(self.csv_file)
                if self.csv_params.header:
                    self.csv_writer.writerow(self.csv_params.header)
                self.open_files.append(self.csv_file)

            if self.video_params:
                self.video_writer = self._open_video(self.video_params)
                self.open_videos.append(self.video_writer)

            if self.plan_view_params:
                self.plan_view_writer = self._open_video(self.plan_view_params)
                self.open_videos.append(self.plan_view_writer)

        except Exception as e:
            # If a file fails to open mid-way, safely close the ones 
            # that already succeeded before raising the error.
            self.__exit__(type(e), e, traceback.format_exc())
            raise e
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self.open_files:
            if not f.closed:
                f.close()
        for cap in self.open_videos:
            cap.release()

    def multiwrite(self, annotated_frame, plan_view, rows):
        if self.csv_writer:
            for row in rows:
                self.csv_writer.writerow(row)
    
        if self.video_writer:
            self.video_writer.write(cv2.resize(annotated_frame, (self.video_params.width, self.video_params.height)))

        if self.plan_view_writer:
            self.plan_view_writer.write(plan_view)


@dataclass
class TWFMRecord:
    final: Path
    working: Path

class TempWorkingFilesManager:
    def __init__(self, final_dir):
        self.registered_files = []
        self.holding_dir = final_dir.with_name(f"WIP{final_dir.name[1:]}")
        self.holding_dir.mkdir(exist_ok=True)

    def _taken(self, x):
        for record in self.registered_files:
            if record.working == x:
                return True
        return False

    def register_and_get_twf(self, final_path):
        working_path = self.holding_dir / final_path.name
        assert not self._taken(working_path)
        self.registered_files.append(TWFMRecord(final=final_path, working=working_path))
        return working_path

    def finalize_and_cleanup(self):
        for record in self.registered_files:
            shutil.move(record.working, record.final)
        self.holding_dir.rmdir()
