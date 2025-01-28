import cv2
import os
# import ffmpeg
import numpy as np
from glob import glob
from natsort import natsorted
from typing import Dict, Iterable, Generator, Tuple
import json
import ipdb
import datetime
from src.utils import params as param_utils

def parse_timestamp(filename):
    # Strip the extension and parse the remaining part as a datetime
    timestamp_str = filename.split('_')[-1].split('.')[0]
    return datetime.datetime.fromtimestamp(int(timestamp_str) / 1e6)

def find_closest_video(folder, anchor_timestamp):
    min_diff = datetime.timedelta(seconds=1)  # Max allowed difference
    closest_video = None

    for filename in os.listdir(folder):
        if filename.endswith('.avi'):
            video_timestamp = parse_timestamp(filename)
            time_diff = abs(video_timestamp - anchor_timestamp)
            
            if time_diff < min_diff:
                min_diff = time_diff
                closest_video = filename
    
    return closest_video, min_diff

class Reader():
    iterator = []

    def __init__(
            self, 
            inp_type: str, 
            path: str, 
            undistort: bool=False, 
            cam_path: str = None,
            cams_to_remove=[], 
            ith: int=0, 
            start_frame: int=0, 
            end_frame: int=-1,
            start_frame_path=None, 
            anchor_camera=None,
            extn: str='jpg',
        ):
        """ith: the ith video in each folder will be processed."""
        self.type = inp_type
        self.ith = ith
        self.frame_count = int(1e9)
        self.start_frames = None
        self.path = path
        self.cams_to_remove = cams_to_remove
        self.to_delete = []
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.extn = extn
        self.undistort = undistort
        self.cameras = param_utils.read_params(cam_path) if self.undistort else None
        
        if self.type == "video":
            self.views = []
            self.streams = {}
            if anchor_camera:
                mp4_list = natsorted(glob(f"{path}/{anchor_camera}/*.avi"))
                if len(mp4_list) > self.ith:
                    self.views.append(natsorted(glob(f"{path}/{anchor_camera}/*.avi"))[self.ith])
            else:
                for cam in os.listdir(path):
                    if 'imu' not in cam and 'mic' not in cam and len(glob(f"{path}/{cam}/*.mp4")) > self.ith:
                        if cam not in cams_to_remove:
                            self.views.append(natsorted(glob(f"{path}/{cam}/*.avi"))[self.ith])
                    if len(self.views) > 0:
                        break
            self.anchor_timestamp = parse_timestamp(self.views[0])
            self.check_timestamp()
            self.init_views()
            if start_frame_path:
                with open(start_frame_path, 'r') as file:
                    start_frames = json.load(file)
                self.start_frames = start_frames
        elif self.type == "image":
            self.views = []
            for cam in os.listdir(path):
                if cam not in cams_to_remove:
                    self.views.append(cam)
            self.init_views()
        else:
            pass

        # Sanity checks
        assert(self.frame_count < int(1e9)), "No frames found"
        if self.frame_count <= 0:
            print("No frames found")
            
        self.cur_frame = 0
    
    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        """ Get next frame (stride 1) from each camera"""
        self.cur_frame = frame_idx
        
        if self.cur_frame == self.frame_count:
            return {}

        frames = {}
        if self.type ==  "video":
            for cam_name, cam_cap in self.streams.items():
                if self.start_frames:
                    start_frame = self.start_frames.get(cam_name, [0, 0])[0]
                else:
                    start_frame = 0
                cam_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + start_frame)
                suc, frame = cam_cap.read()
                if not suc:
                    raise RuntimeError(f"Couldn't retrieve frame from {cam_name}")
                if self.undistort:
                    idx = np.where(self.cameras[:]['cam_name']==cam_name)[0][0]
                    cam = self.cameras[idx]
                    K, dist = param_utils.get_intr(cam)
                    new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                    frame = param_utils.undistort_image(K, new_K, dist, frame)
                frames[cam_name] = frame
        elif self.type == "image":
            for view in self.views:
                frame_path = os.path.join(self.path, view, f"{self.cur_frame:08d}.{self.extn}")

                if self.extn == "png":
                    frame = cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                else:
                    frame = cv2.imread(frame_path)

                if self.undistort:
                    idx = np.where(self.cameras[:]['cam_name']==view)[0][0]
                    cam = self.cameras[idx]
                    K, dist = param_utils.get_intr(cam)
                    new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                    frame = param_utils.undistort_image(K, new_K, dist, frame)
                frames[view] = frame
        return frames

    def check_timestamp(self):
        for cam in os.listdir(self.path):
            if 'imu' not in cam and 'mic' not in cam and cam not in self.cams_to_remove and cam not in self.views[0]:
                closest_file, time_diff = find_closest_video(f"{self.path}/{cam}", self.anchor_timestamp)
                if closest_file:
                    self.views.append(f"{self.path}/{cam}/{closest_file}")
                else:
                    self.to_delete.append(cam.split('/')[-1].rsplit('_', 0)[0])
            
    def reinit(self):
        """ Reinitialize the reader """
        if self.type == "video":
            self.release_videos()
            self.init_views()

        self.cur_frame = 0

    def init_views(self):
        """ Create video captures for each video
                ith: the ith video in each folder will be processed."""
        if self.type == "video":
            for vid in self.views:
                cap = cv2.VideoCapture(vid)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_count = min(self.frame_count, frame_count)
                cam_name = vid.split("/")[-2]
                self.streams[cam_name] = cap
                
            self.frame_count -= 5 # To account for the last few frames that are corrupted
        elif self.type == "image":
            for vid in self.views:
                if self.end_frame > 0:
                    frame_count = self.end_frame - self.start_frame
                else:
                    frame_count = len(glob(os.path.join(self.path, self.views[0], f"*.{self.extn}")))
                self.frame_count = min(self.frame_count, frame_count)
        else:
            pass

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

    def get_image_frames(self, view, undistort=False, intr=None, dist_intr=None, dist=None, target_fps=None, target_frames=None):
        orig_imgs = []
        im_names = []
        
        frames = glob(os.path.join(self.path, view, "*.jpg"))

        if self.end_frame > 0:
            frames = frames[self.start_frame:self.end_frame]
        else:
            frames = frames[self.start_frame:]

        for frame_path in frames:
            frame = cv2.imread(frame_path)
            
            if undistort:
                idx = np.where(self.cameras[:]['cam_name']==view)[0][0]
                cam = self.cameras[idx]
                K, dist = param_utils.get_intr(cam)
                new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                frame = param_utils.undistort_image(K, new_K, dist, frame)
            
            orig_imgs.append(frame)
            im_names.append(os.path.basename(frame_path))
        H, W, _ = frame.shape

        return im_names, orig_imgs, H, W