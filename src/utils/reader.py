import cv2
import os
import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple
import json
import datetime
from src.utils import params as param_utils


class Reader():
    iterator = []

    def __init__(
            self, inp_type: str, path: str, undistort: bool=False, cam_path: str=None, cams_to_remove = [], ith: int=0, start_frame: int=0, sample_fps: int=None, frame_count: int=None):
        """ith: the ith video in each folder will be processed."""
        self.type = inp_type
        self.ith = ith
        self.path = path
        self.to_delete = []
        self.undistort = undistort
        self.start_frame = start_frame
        self.sample_fps = sample_fps
        self.frame_count = int(1e9)
        self.target_frames = frame_count

        if self.undistort: 
            assert (cam_path is not None)
            self.cameras = param_utils.read_params(cam_path)
        else: 
            self.cameras = None

        if self.type == "video":
            self.streams = {}
            self.vids = []
            for cam in os.listdir(path):
                if 'imu' not in cam and len(glob(f"{path}/{cam}/*.avi")) > self.ith:
                    if len(cams_to_remove) !=0: 
                        if cam not in cams_to_remove: 
                            self.vids.append(natsorted(glob(f"{path}/{cam}/*.avi"))[self.ith])
                    else: 
                        self.vids.append(natsorted(glob(f"{path}/{cam}/*.avi"))[self.ith])
                        
            self.init_videos()
        else:
            pass

        self.cur_frame = start_frame

        # Sanity checks
        if (self.target_frames <=0 or self.target_frames > 1e8): 
            raise ValueError ("frame count is more than 1e8, no videos loaded!")
    
    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        self.cur_frame = frame_idx
        
        if self.cur_frame == self.target_frames:
            return {}

        frames = {}
        for cam_name, cam_cap in self.streams.items():
            frame_num = round(frame_idx * (self.original_frame_count - 1) / (self.frame_count - 1))
            cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + self.start_frame)
            suc, frame = cam_cap.read()
            if not suc:
                raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")

            if self.undistort: 
                idx = np.where(self.cameras[:]['cam_name']==cam_name)[0][0]
                cam = self.cameras[idx]
                assert (cam['cam_name'] == cam_name)
                extr = param_utils.get_extr(cam)
                K, dist = param_utils.get_intr(cam)
                new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                frame = param_utils.undistort_image(K, new_K, dist, frame)
            frames[cam_name] = frame
        
        return frames

    def reinit(self):
        """ Reinitialize the reader """
        if self.type == "video":
            self.release_videos()
            self.init_videos()

        self.cur_frame = 0

    def init_videos(self):
        """ Create video captures for each video
                ith: the ith video in each folder will be processed."""
        for vid in self.vids:
            cap = cv2.VideoCapture(vid)

            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(ffmpeg.probe(vid, cmd="ffprobe")["streams"][0]["nb_frames"])
            self.original_frame_count = frame_count

            # Get the frame count if you are subsampling the video
            if self.sample_fps is not None:
                assert self.sample_fps <= orig_fps, "Target FPS must be less than or equal to original FPS."
                duration_seconds = frame_count / orig_fps
                frame_count = int(duration_seconds * self.sample_fps)

            self.frame_count = min(self.frame_count, frame_count)
            cam_name = os.path.basename(vid).split(".")[0]
            self.streams[cam_name] = cap
            
            # If number of target frames are not specified, set it to the frame count
            if not self.target_frames:
                self.target_frames = frame_count

    def release_videos(self):
        for cap in self.streams.values():
            cap.release()
    
    def __call__(self, frames: Iterable[int]=[]):
        # Sort the frames so that we access them in order
        frames = sorted(frames)
        self.iterator = frames
        
        for frame_idx in frames:
            frame = self._get_next_frame(frame_idx)
            if not frame:
                break
            
            yield frame, self.cur_frame

        # Reinitialize the videos
        self.reinit()

    def __len__(self): 
        return len(self.vids)