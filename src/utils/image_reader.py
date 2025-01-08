import os
import cv2
import numpy as np

from glob import glob
from typing import Dict, Iterable
from src.utils import params as param_utils

class Reader():

    def __init__(
            self,
            path: str,
            undistort: bool = False,
            cam_path: str = None,
            cams_to_remove=[],
            start_frame: int = 0,
            end_frame: int = None,
            extn: str = "jpg"
    ):
        self.path = path
        self.undistort = undistort
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.cameras = param_utils.read_params(cam_path) if self.undistort else None
        self.frame_count = int(1e9)
        self.extn = extn

        self.views = []
        for cam in os.listdir(path):
            if cam not in cams_to_remove:
                self.views.append(cam)
        self.init_views()
        
        self.curr_frame = start_frame
        if self.frame_count <= 0 or self.frame_count > 1e8:
            raise ValueError("frame count is more than 1e8, no views loaded!")
        

    def init_views(self):
        if self.end_frame:
            frame_count = self.end_frame - self.start_frame
        else:
            frame_count = len(glob(os.path.join(self.path, self.views[0], f"*.{self.extn}")))
        self.frame_count = min(self.frame_count, frame_count)

    
    def _get_next_frame(self, frame_idx) -> Dict[str, np.ndarray]:
        self.curr_frame = frame_idx

        if self.curr_frame == self.end_frame:
            return {}

        frames = {}
        for view in self.views:
            idx = np.where(self.cameras[:]['cam_name']==view)[0][0]
            cam = self.cameras[idx]
            frame_path = os.path.join(self.path, view, f"{self.curr_frame:08d}.{self.extn}")
            frame = cv2.cvtColor(cv2.imread(frame_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)

            if self.undistort:
                K, dist = param_utils.get_intr(cam)
                new_K, roi = param_utils.get_undistort_params(K, dist, (frame.shape[1], frame.shape[0]))
                frame = param_utils.undistort_image(K, new_K, dist, frame)
            frames[view] = frame
        
        return frames
    

    def __call__(self, frames: Iterable[int]=[]):
        frames = sorted(frames)
        self.iterator = frames

        for frame_idx in frames:
            frame = self._get_next_frame(frame_idx)
            if not frame:
                break
                
            yield frame, self.curr_frame

    def __len__(self):
        return len(self.cameras)
    

    def get_frames(self, view, undistort=False, intr=None, dist_intr=None, dist=None, target_fps=None, target_frames=None):
        orig_imgs = []
        im_names = []

        frames = glob(os.path.join(self.path, view, "*.jpg"))

        if self.end_frame:
            frames = frames[self.start_frame:self.end_frame]
        else:
            frames = frames[self.start_frame:]

        for frame_path in frames:
            frame = cv2.imread(frame_path)
            if undistort:
                frame = cv2.undistort(frame, intr, dist, None, dist_intr)
            
            orig_imgs.append(frame)
            im_names.append(os.path.basename(frame_path))
        H, W, _ = frame.shape

        return im_names, orig_imgs, H, W