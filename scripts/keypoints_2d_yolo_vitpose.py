from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import ujson
import time
from ultralytics import YOLO, checks

from typing import Dict, Optional
from collections import defaultdict

import sys
sys.path.append(".")
from src.utils.image_reader import Reader
from src.utils.video_handler import frame_preprocess
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.vitpose_wrapper import ViTPoseModel
from src.hamer_wrapper import HAMER_CKPT_PATH, ViTDetDataset, recursive_clear

# ------------------------------ Alpha Pose Helpers ------------------------------ #


def process_hand_keypoints_batch(keypoints, validity_threshold, min_valid_keypoints):
    valid_mask = keypoints[:, :, 2] > validity_threshold
    valid_counts = np.sum(valid_mask, axis=1)
    
    # Initialize storage for bounding boxes and reshape keypoints
    bboxes = np.zeros((keypoints.shape[0], 4))
    reshaped_keypoints = np.zeros((keypoints.shape[0], keypoints.shape[1] * keypoints.shape[2]))

    # Process only those with enough valid keypoints
    sufficient_valid = valid_counts > min_valid_keypoints
    if np.any(sufficient_valid):
        filtered_keypoints = np.where(valid_mask[:,:,None], keypoints, np.nan)  # Replace invalid keypoints with NaN for min/max operations
        bboxes[sufficient_valid, 0] = np.nanmin(filtered_keypoints[sufficient_valid, :, 0], axis=1)
        bboxes[sufficient_valid, 1] = np.nanmin(filtered_keypoints[sufficient_valid, :, 1], axis=1)
        bboxes[sufficient_valid, 2] = np.nanmax(filtered_keypoints[sufficient_valid, :, 0], axis=1)
        bboxes[sufficient_valid, 3] = np.nanmax(filtered_keypoints[sufficient_valid, :, 1], axis=1)
        reshaped_keypoints[sufficient_valid] = keypoints[sufficient_valid].reshape(-1, keypoints.shape[1] * keypoints.shape[2])
        
    return bboxes, reshaped_keypoints, valid_counts

def prorcess_all_hamerposes(hamer_batch, hamer_out, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f):
    boxes = hamer_batch['boxes']
    box_center = hamer_batch["box_center"].float()
    box_size = hamer_batch["box_size"].float()
    
    batch_size = hamer_batch['img'].shape[0]
    for n in range(batch_size):
        is_right = hamer_batch['right'][n]
        pred_keypoints_2d = hamer_out['pred_keypoints_2d'][n, :, :].squeeze()
        multiplier = (2*is_right-1)
        pred_keypoints_2d[:, 0] = pred_keypoints_2d[:, 0] * multiplier
        joints = pred_keypoints_2d.detach().cpu() * box_size[n] + box_center[n, :]

        joints = np.hstack((joints.numpy(), np.ones((21, 1)))).reshape(-1).tolist()
        
        box = boxes[n, :].tolist()
        if is_right:
            ujson.dump(joints, kps_right_f)
            kps_right_f.write('\n')

            ujson.dump(box, bbx_right_f)
            bbx_right_f.write('\n')
        else:
            ujson.dump(joints, kps_left_f)
            kps_left_f.write('\n')

            ujson.dump(box, bbx_left_f)
            bbx_left_f.write('\n')
            
    
def process_all_vitposes_for_hamer(pred_poses, frame_buffer):
    all_processed_bbox = []
    is_right = []
    for pred_pose in pred_poses:
        processed_pose = process_all_vitposes(pred_pose)
        all_processed_bbox.append(processed_pose['left_bbox'])
        all_processed_bbox.append(processed_pose['right_bbox'])
        is_right.extend([0, 1])
    
    all_processed_bbox_array = np.array(all_processed_bbox)
    is_right_array = np.array(is_right)
    frame_buffer = np.array(frame_buffer)
    repeated_frame_buffer = frame_buffer[np.repeat(np.arange(len(frame_buffer)), 2)]
    return all_processed_bbox_array, is_right_array, repeated_frame_buffer
    
def process_all_vitposes(pred_poses, kps_left_f=None, bbx_left_f=None, kps_right_f=None, bbx_right_f=None):                    
    if len(pred_poses) == 0:
        left_bbox = [0.0] * 4
        right_bbox = [0.0] * 4
        left_keyp = [0.0] * (21 * 3)
        right_keyp = [0.0] * (21 * 3)
    else:
        keypoints_all = pred_poses
        
        # Split into left and right hands
        left_hand_keyps = keypoints_all[:, -42:-21, :]
        right_hand_keyps = keypoints_all[:, -21:, :]

        # Process each hand
        left_bboxes, left_keyps, left_valid_counts = process_hand_keypoints_batch(left_hand_keyps, 0.5, 3)
        right_bboxes, right_keyps, right_valid_counts = process_hand_keypoints_batch(right_hand_keyps, 0.5, 3)

        # To identify best left and right hands, consider a criteria, e.g., max valid keypoints
        best_left_index = np.argmax(left_valid_counts)
        best_right_index = np.argmax(right_valid_counts)

        left_bbox = left_bboxes[best_left_index].tolist()
        right_bbox = right_bboxes[best_right_index].tolist()
        left_keyp = left_keyps[best_left_index].tolist()
        right_keyp = right_keyps[best_right_index].tolist()
    
    # Writting
    if kps_left_f:
        ujson.dump(left_keyp, kps_left_f)
        kps_left_f.write('\n')

        ujson.dump(left_bbox, bbx_left_f)
        bbx_left_f.write('\n')

        ujson.dump(right_keyp, kps_right_f)
        kps_right_f.write('\n')

        ujson.dump(right_bbox, bbx_right_f)
        bbx_right_f.write('\n')
    else:
        return {
            'left_keyp': left_keyp,
            'left_bbox': left_bbox,
            'right_keyp': right_keyp,
            'right_bbox': right_bbox,
        }
    

def process_all_yolo_results(results, bboxes_buffer, im_h, im_w, box_score_threshold=0.2, padding=5):

    for result in results:
        if len(result) == 0:
            pred_bboxes_scores = []
        else:
            valid_idx = (result.boxes.cls==0) & (result.boxes.conf > box_score_threshold)
            pred_bboxes=result.boxes.xyxy[valid_idx].cpu().numpy()
            padded_bboxes = np.copy(pred_bboxes)
            padded_bboxes[:, 0:2] -= padding  # x_min - 10
            padded_bboxes[:, 2:] += padding  # x_max + 10
            padded_bboxes[:, 0] = np.clip(padded_bboxes[:, 0], 0, im_h)  # x_min
            padded_bboxes[:, 1] = np.clip(padded_bboxes[:, 1], 0, im_w) # y_min
            padded_bboxes[:, 2] = np.clip(padded_bboxes[:, 2], 0, im_h)  # x_max
            padded_bboxes[:, 3] = np.clip(padded_bboxes[:, 3], 0, im_w) # y_max
            pred_scores=result.boxes.conf[valid_idx].cpu().numpy()
            pred_bboxes_scores = np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)
        bboxes_buffer.append(pred_bboxes_scores)


def main():
    parser = argparse.ArgumentParser(description='2D Keypoint Detection')
    add_common_args(parser)
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--box_score_threshold', type=float, default=0.2, help='Confidence Threshold for BBX Detection')
    parser.add_argument('--yolo_model', type=str, default='yolov9c.pt', help='YOLO Model for BBX Detection')
    parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
    parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
    parser.add_argument('--use_hamer', type=bool, default=False, help='YOLO -> ViTPose -> Hamer pipeline')
    args = parser.parse_args()

    # Setup HaMeR model
    device = torch.device('cuda')
    cpm = ViTPoseModel(device)
    model = YOLO(args.yolo_model)

    if args.use_hamer:
        from hamer.models import load_hamer
        from hamer.utils import recursive_to
        hamer_model, hamer_model_cfg = load_hamer(HAMER_CKPT_PATH)
        hamer_model = hamer_model.to(device)
        hamer_model.eval()

    input_path = os.path.join(args.root_dir, args.seq_path, "data", "image")

    if args.use_optim_params:
        params_txt = "optim_params.txt"
    else:
        params_txt = "params.txt"

    params_path = os.path.join(args.out_dir, params_txt)
    params = param_utils.read_params(params_path)
    cam_names = list(params[:]["cam_name"])
    removed_camera_path = os.path.join(args.out_dir, 'ignore_camera.txt')
    if os.path.isfile(removed_camera_path):
        with open(removed_camera_path) as file:
            ignored_cameras = [line.rstrip() for line in file]
    else:
        ignored_cameras = None
    cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam, ignored_cameras=ignored_cameras)

    for cam in cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)
    cam_mapper = map_camera_names(input_path, cam_names)

    if args.ith == -1:
        total_video_idxs = 0
        max_folder_id = 0
        for fid, folder in enumerate(os.listdir(input_path)):
            if 'cam' in folder and folder not in cams_to_remove:
                length = len([file for file in os.listdir(os.path.join(input_path, folder)) if file.endswith('.mp4')])
                if length > total_video_idxs:
                    total_video_idxs = length
                    max_folder_id = fid
                    anchor_camera_by_length = os.listdir(input_path)[fid]
                # remove any cameras not in the param file
                if folder not in cams_to_remove:
                    cams_to_remove.append(folder)
        if args.start > 0:
            if args.end > 0:
                selected_vid_idxs = list(range(args.start, args.end))
            else:
                selected_vid_idxs = list(range(args.start, total_video_idxs))
        else:
            if args.end > 0:
                selected_vid_idxs = list(range(args.end))
            else:
                selected_vid_idxs = list(range(total_video_idxs))
    else:       
        selected_vid_idxs = [args.ith]
        # remove any cameras that are not in the params file
        for folder in os.listdir(input_path):
            if 'cam' in folder and folder not in cam_names and folder not in cams_to_remove:
                cams_to_remove.append(folder)
    
    for selected_vid_idx in selected_vid_idxs:
        print(f'Video ID {selected_vid_idx}...')
        
        output_kps_left_path = f'{args.out_dir}/keypoints_2d/left/{selected_vid_idx:03d}'
        output_bbx_left_path = f'{args.out_dir}/bboxes/left/{selected_vid_idx:03d}'
        output_kps_right_path = f'{args.out_dir}/keypoints_2d/right/{selected_vid_idx:03d}'
        output_bbx_right_path = f'{args.out_dir}/bboxes/right/{selected_vid_idx:03d}'
        os.makedirs(output_kps_left_path, exist_ok=True)
        os.makedirs(output_bbx_left_path, exist_ok=True)
        os.makedirs(output_kps_right_path, exist_ok=True)
        os.makedirs(output_bbx_right_path, exist_ok=True)

        # Get files to process
        reader = Reader(input_path, undistort=True, cam_path=params_path, cams_to_remove=cams_to_remove, start_frame=args.start, end_frame=args.end)

        if reader.frame_count <= 0:
            continue
        
        print("Total Views:", len(reader.views))
        print("Total frames:", reader.frame_count)
        
        print("Reader Length", len(reader.views))

        # Detect 2D Keypoints for all valid views
        time_list = []
        for v_idx, input_view in tqdm(enumerate(reader.views), total=len(reader.views)):
            im_names, orig_imgs, im_h, im_w = reader.get_frames(input_view)
            
            output_kps_left_file_path = f"{output_kps_left_path}/{input_view}.jsonl"
            output_bbx_left_file_path = f"{output_bbx_left_path}/{input_view}.jsonl"
            output_kps_right_file_path = f"{output_kps_right_path}/{input_view}.jsonl"
            output_bbx_right_file_path = f"{output_bbx_right_path}/{input_view}.jsonl"                
            
            start_time = time.time()
            with open(output_kps_left_file_path, 'w') as kps_left_f, open(output_bbx_left_file_path, 'w') as bbx_left_f, open(output_kps_right_file_path, 'w') as kps_right_f, open(output_bbx_right_file_path, 'w') as bbx_right_f:
                frame_buffer, bboxes_buffer = [], []
                for im_name, frame in zip(im_names, orig_imgs):
                    frame_buffer.append(frame)
                    if len(frame_buffer) == args.batch_size:
                        # Detect humans in image
                        with torch.no_grad():
                            results = model(frame_buffer, verbose=False, stream=True)
                        process_all_yolo_results(results, bboxes_buffer, im_h, im_w, args.box_score_threshold, padding=0)
                        
                        # Detect human keypoints for each person
                        with torch.no_grad():        
                            pred_poses = cpm.predict_pose_batch(
                                frame_buffer,
                                bboxes_buffer
                            )

                        if args.use_hamer:
                            boxes, right, repeated_frame_buffer = process_all_vitposes_for_hamer(pred_poses, frame_buffer)
                            hamer_dataset = ViTDetDataset(hamer_model_cfg, repeated_frame_buffer, boxes, right, rescale_factor=2.0, device=device)
                            hamer_dataloader = torch.utils.data.DataLoader(hamer_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
                            for hamer_batch in hamer_dataloader:
                                with torch.no_grad():
                                    hamer_out = hamer_model(hamer_batch)
                                    recursive_clear(hamer_batch)
                                prorcess_all_hamerposes(hamer_batch, hamer_out, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f)                            
                        else:
                            for pred_pose in pred_poses:
                                processed_pose = process_all_vitposes(pred_pose, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f)
                            
                        frame_buffer, bboxes_buffer = [], []

                if len(frame_buffer) > 0:
                    # Detect humans in image
                    results = model(frame_buffer, verbose=False, stream=True)
                    process_all_yolo_results(results, bboxes_buffer, im_h, im_w, args.box_score_threshold, padding=5)
                    
                    # Detect human keypoints for each person               
                    pred_poses = cpm.predict_pose_batch(
                        frame_buffer,
                        bboxes_buffer
                    )

                    if args.use_hamer:
                        boxes, right, repeated_frame_buffer = process_all_vitposes_for_hamer(pred_poses, frame_buffer)
                        hamer_dataset = ViTDetDataset(hamer_model_cfg, repeated_frame_buffer, boxes, right, rescale_factor=2.0, device=device)
                        hamer_dataloader = torch.utils.data.DataLoader(hamer_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)
                        for hamer_batch in hamer_dataloader:
                            start_time = time.time()
                            with torch.no_grad():
                                hamer_out = hamer_model(hamer_batch)
                                recursive_clear(hamer_batch)
                            prorcess_all_hamerposes(hamer_batch, hamer_out, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f) 
                                
                    else:
                        for pred_pose in pred_poses:
                            processed_pose = process_all_vitposes(pred_pose, kps_left_f, bbx_left_f, kps_right_f, bbx_right_f)
            time_list.append(time.time() - start_time)            
        
if __name__ == '__main__':
    main()
