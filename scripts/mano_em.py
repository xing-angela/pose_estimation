import os
import cv2
import sys
import json
import ujson
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(".")
# from src.utils.reader_v2 import Reader
# from src.utils.reader import Reader
from src.utils.image_reader import Reader
import src.utils.params as param_utils
from src.utils.parser import add_common_args
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.utils.easymocap_utils import vis_smpl, projectN3, vis_repro, load_model
from src.utils.filter import apply_one_euro_filter_2d, apply_one_euro_filter_3d
from src.utils.video_handler import create_video_writer, convert_video_ffmpeg
sys.path.append("./third-party/EasyMocap")
from easymocap.mytools import Timer
from easymocap.dataset import CONFIG
from easymocap.pipeline import smpl_from_keypoints3d2d, smpl_from_keypoints3d
from easymocap.smplmodel import check_keypoints, select_nf

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# parser = load_parser()
parser = argparse.ArgumentParser("Mano Fitting Argument Parser")
add_common_args(parser)
parser.add_argument("--use_optim_params", action="store_true")
parser.add_argument("--to_smooth", action="store_true", help="Whether to temporally smoothing the result")
parser.add_argument("--use_filtered", action="store_true", help="Whether to use only filtered keypoints (binned)")
parser.add_argument('--remove_side_cam', type=bool, default=True, help='Remove Side Cameras')
parser.add_argument('--remove_bottom_cam', type=bool, default=True, help='Remove Bottom Cameras')
# Easy Mocap Arguments
parser.add_argument('--body', type=str, default='body25', choices=['body15', 'body25', 'h36m', 'bodyhand', 'bodyhandface', 'handl', 'handr', 'handlr', 'total'])
parser.add_argument('--model', type=str, default='smpl', choices=['smpl', 'smplh', 'smplx', 'manol', 'manor'])
parser.add_argument('--gender', type=str, default='neutral', choices=['neutral', 'male', 'female'])
parser.add_argument('--save_origin', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--opts', help="Modify config options using the command-line", 
    default={}, nargs='+')
# optimization control
recon = parser.add_argument_group('Reconstruction control')
recon.add_argument('--robust3d', action='store_true')
# visualization
output = parser.add_argument_group('Output control')
output.add_argument('--write_smpl_full', action='store_true')
output.add_argument('--vis_2d_repro', action='store_true')
output.add_argument('--vis_3d_repro', action='store_true')
output.add_argument('--vis_smpl', action='store_true')
output.add_argument('--save_frame', action='store_true')
output.add_argument('--save_mesh', action='store_true')
args = parser.parse_args()



# -------------------- Visualization Functions -------------------- #


# Loads the camera parameters
if args.use_optim_params:
    params_txt = "optim_params.txt"
else:
    params_txt = "params.txt"

base_path = os.path.join(args.root_dir)
image_dir = os.path.join(base_path, args.seq_path, "data", "image")
output_path = args.out_dir
params_path = os.path.join(output_path, params_txt)

params = param_utils.read_params(params_path)
cam_names = list(params[:]["cam_name"])
removed_camera_path = os.path.join(output_path, 'ignore_camera.txt')
if os.path.isfile(removed_camera_path):
    with open(removed_camera_path) as file:
        ignored_cameras = [line.rstrip() for line in file]
else:
    ignored_cameras = None
cams_to_remove = removed_cameras(remove_side=args.remove_side_cam, remove_bottom=args.remove_bottom_cam, ignored_cameras=ignored_cameras)
for cam in cams_to_remove:
    if cam in cam_names:
        cam_names.remove(cam)

if args.ith == -1:
    total_video_idxs = 0
    max_folder_id = 0
    for fid, folder in enumerate(os.listdir(image_dir)):
        if 'cam' in folder and folder not in cams_to_remove:
            length = len([file for file in os.listdir(os.path.join(image_dir, folder)) if file.endswith('.mp4')])
            if length > total_video_idxs:
                total_video_idxs = length
                max_folder_id = fid
                anchor_camera_by_length = os.listdir(image_dir)[fid]
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

for selected_vid_idx in selected_vid_idxs:
    print(f'Video ID {selected_vid_idx}...')
    # reader = Reader("video", image_dir, cams_to_remove=cams_to_remove, ith=selected_vid_idx, anchor_camera=anchor_camera_by_length if args.ith==-1 else args.anchor_camera)
    # reader = Reader("video", image_dir, cams_to_remove=cams_to_remove, undistort=True, cam_path=params_path, sample_fps=args.sample_fps, frame_count=args.frame_count)
    reader = Reader(image_dir, undistort=True, cam_path=params_path, cams_to_remove=cams_to_remove, start_frame=args.start, end_frame=args.end)

    if reader.frame_count <= 0:
        continue
    
    keypoints2d_dir_right = os.path.join(output_path, "keypoints_2d", "right", str(selected_vid_idx).zfill(3))
    keypoints2d_dir_left = os.path.join(output_path, "keypoints_2d", "left",  str(selected_vid_idx).zfill(3))
    bboxes_dir_right = os.path.join(output_path, "bboxes", "right",  str(selected_vid_idx).zfill(3))
    bboxes_dir_left = os.path.join(output_path, "bboxes", "left",  str(selected_vid_idx).zfill(3))
    keypoints3d_dir = os.path.join(output_path, "keypoints_3d", str(selected_vid_idx).zfill(3))
    keypt3d_file_left = os.path.join(keypoints3d_dir, "left.jsonl")
    keypt3d_file_right = os.path.join(keypoints3d_dir, "right.jsonl")

    # loads the selected frames
    if args.use_filtered:
        chosen_path_left = os.path.join(keypoints3d_dir, f"chosen_frames_left.json")
        chosen_path_right = os.path.join(keypoints3d_dir, f"chosen_frames_right.json")
        with open(chosen_path_right, "r") as f:
            chosen_frames_right = set(json.load(f))
        with open(chosen_path_left, "r") as f:
            chosen_frames_left = set(json.load(f))
        chosen_frames =list(set(chosen_frames_right | chosen_frames_left))
    else:
        chosen_frames = range(args.start, args.end, args.stride)

    chosen_frames = sorted(chosen_frames)
    print(f"Total valid frames {len(chosen_frames)}/{reader.frame_count}")
    
    cam_mapper = map_camera_names(keypoints2d_dir_right, cam_names)
    intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, reader.views, cam_mapper, easymocap_format=True)

    # loads 2d & 3d keypoints
    all_keypoints2d_left, all_keypoints2d_right = [], []
    all_bboxes_left, all_bboxes_right = [], []
    for cam in reader.views:
        if cam in cam_mapper:
            keypoints2d_left = []
            keypoints2d_right = []
            bboxes_left = []
            bboxes_right = []
            ap_keypoints_path_left = os.path.join(keypoints2d_dir_left, f"{cam_mapper[cam]}.jsonl")
            ap_keypoints_path_right = os.path.join(keypoints2d_dir_right, f"{cam_mapper[cam]}.jsonl")
            bboxes_path_left = os.path.join(bboxes_dir_right, f"{cam_mapper[cam]}.jsonl")
            bboxes_path_right = os.path.join(bboxes_dir_left, f"{cam_mapper[cam]}.jsonl")
            with open(ap_keypoints_path_left, "r") as fl, open(ap_keypoints_path_right, "r") as fr, open(bboxes_path_left, "r") as fbl, open(bboxes_path_right, "r") as fbr:
                for l_idx, (linel, liner, linebl, linebr) in enumerate(zip(fl, fr, fbl, fbr)):
                    if l_idx in chosen_frames:
                        keypoints2d_left.append(np.array(ujson.loads(linel)).reshape(-1, 3))
                        keypoints2d_right.append(np.array(ujson.loads(liner)).reshape(-1, 3))
                        bboxes_left.append(np.array(ujson.loads(linebl) + [1.0]))
                        bboxes_right.append(np.array(ujson.loads(linebr) + [1.0]))
            all_keypoints2d_left.append(np.asarray(keypoints2d_left))
            all_keypoints2d_right.append(np.asarray(keypoints2d_right))
            all_bboxes_left.append(np.asarray(bboxes_left))
            all_bboxes_right.append(np.asarray(bboxes_right))
            
    all_keypoints2d_left = np.asarray(all_keypoints2d_left)
    all_keypoints2d_right = np.asarray(all_keypoints2d_right)
    all_bboxes_left = np.asarray(all_bboxes_left)
    all_bboxes_right = np.asarray(all_bboxes_right)
    all_keypoints2d_left = np.swapaxes(all_keypoints2d_left, 0, 1)
    all_keypoints2d_right = np.swapaxes(all_keypoints2d_right, 0, 1)
    all_bboxes_left = np.swapaxes(all_bboxes_left, 0, 1)
    all_bboxes_right = np.swapaxes(all_bboxes_right, 0, 1)
    for nf in range(all_keypoints2d_left.shape[0]):
        all_keypoints2d_left[nf, :, :, :2] = param_utils.undistort_points(all_keypoints2d_left[nf, :, :, :2], cameras)
    for nf in range(all_keypoints2d_right.shape[0]):
        all_keypoints2d_right[nf, :, :, :2] = param_utils.undistort_points(all_keypoints2d_right[nf, :, :, :2], cameras)
        
    keypoints3d_right, keypoints3d_left = [], []
    with open(keypt3d_file_left, "r") as fl, open(keypt3d_file_right, "r") as fr:
        for l_idx, (linel, liner) in enumerate(zip(fl, fr)):
            if l_idx in chosen_frames:
                keypoints3d_left.append(np.array(ujson.loads(linel)).reshape(-1, 4))
                keypoints3d_right.append(np.array(ujson.loads(liner)).reshape(-1, 4))
    keypoints3d_left = np.asarray(keypoints3d_left)
    keypoints3d_right = np.asarray(keypoints3d_right)
    
    # loads the mano model
    with Timer('Loading {}, {}'.format(args.model, args.gender), not False):
        body_model_right = load_model(gender=args.gender, model_type=args.model, model_path="data/smplx", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)

    with Timer('Loading {}, {}'.format(args.model, args.gender), not False):
        body_model_left = load_model(gender=args.gender, model_type=args.model.replace('r', 'l'), model_path="data/smplx", num_pca_comps=6, use_pose_blending=True, use_shape_blending=True, use_pca=False, use_flat_mean=False)
        # body_model_left = load_model(gender=args.gender, model_type=args.model, model_path="data/smplx", num_pca_comps=6, use_pca=True, use_flat_mean=True)

    # fits the mano model
    dataset_config = CONFIG[args.body]

    if len(keypoints3d_right.shape) == 3:
        if args.to_smooth:
            print('Smoothing Keypoints 3D...')
            keypoints3d_left[:3] = apply_one_euro_filter_3d(keypoints3d_left[:3], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            keypoints3d_right[:3] = apply_one_euro_filter_3d(keypoints3d_right[:3], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)

        weight_pose = {
            'k3d': 1e2, 'k2d': 2e-3,
            'reg_poses': 1e-3, 'smooth_body': 1e2, 'smooth_poses': 1e2,
        }

        # params_right = smpl_from_keypoints3d2d(body_model_right, keypoints3d_right, all_keypoints2d_right, all_bboxes_right, projs, 
        #     config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
        params_right = smpl_from_keypoints3d(body_model_right, keypoints3d_right, 
            config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
        # params_left = smpl_from_keypoints3d2d(body_model_left, keypoints3d_left, all_keypoints2d_left, all_bboxes_left, projs, 
        #     config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)
        params_left = smpl_from_keypoints3d(body_model_left, keypoints3d_left, 
            config=dataset_config, args=args, weight_shape={'s3d': 1e5, 'reg_shapes': 5e3}, weight_pose=weight_pose)

        if args.to_smooth:
            print('Smoothing Manos...')
            params_right['Rh'] = apply_one_euro_filter_2d(params_right['Rh'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_left['Rh'] = apply_one_euro_filter_2d(params_left['Rh'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_right['Th'] = apply_one_euro_filter_2d(params_right['Th'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_left['Th'] = apply_one_euro_filter_2d(params_left['Th'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_right['poses'] = apply_one_euro_filter_2d(params_right['poses'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)
            params_left['poses'] = apply_one_euro_filter_2d(params_left['poses'], mincutoff = 0.5, beta = 0.0, dcutoff = 1.0)

        # save parameters
        manos_params = {}
        params_left_list = {}
        params_right_list = {}
        for key in params_left:
            params_left_list[key] = params_left[key].tolist()
        for key in params_right:
            params_right_list[key] = params_right[key].tolist()
        manos_params['left'] = params_left_list
        manos_params['right'] = params_right_list
        outhand_mano_params_path = f'{output_path}/params/{str(selected_vid_idx).zfill(3)}.json'
        os.makedirs(os.path.dirname(outhand_mano_params_path), exist_ok=True)
        with open(outhand_mano_params_path, "w") as f:
            ujson.dump(manos_params, f)
            
        # visualize model
        if args.vis_smpl or args.save_mesh or args.vis_2d_repro or args.vis_3d_repro:
            import trimesh

            if args.vis_smpl:
                if not args.save_frame:
                    os.makedirs(f'{output_path}/mano', exist_ok=True)
                    outhand_mano_path = f'{output_path}/mano/{str(selected_vid_idx).zfill(3)}.mp4'
                else:
                    outhand_mano_path = f'{output_path}/mano/{str(selected_vid_idx).zfill(3)}'
                    os.makedirs(outhand_mano_path, exist_ok=True)
            if args.vis_2d_repro:
                if not args.save_frame:
                    os.makedirs(f'{output_path}/repro_2d', exist_ok=True)
                    outhand_2d_path = f'{output_path}/repro_2d/{str(selected_vid_idx).zfill(3)}.mp4'
                else:
                    outhand_2d_path = f'{output_path}/repro_2d/{str(selected_vid_idx).zfill(3)}'
                    os.makedirs(outhand_2d_path, exist_ok=True)
            if args.vis_3d_repro:
                if not args.save_frame:
                    os.makedirs(f'{output_path}/repro_3d', exist_ok=True)
                    outhand_3d_path = f'{output_path}/repro_3d/{str(selected_vid_idx).zfill(3)}.mp4'
                else:
                    outhand_3d_path = f'{output_path}/repro_3d/{str(selected_vid_idx).zfill(3)}'
                    os.makedirs(outhand_3d_path, exist_ok=True)
            nf = 0
            
            for abs_idx, (frames, idx) in tqdm(enumerate(reader(chosen_frames)), total=len(chosen_frames)):
                images = []
                c_idx = 0
                for cam in reader.views:
                    if cam in cam_mapper:
                        image = frames[cam_mapper[cam]]
                        if args.undistort:
                            image = param_utils.undistort_image(intrs[c_idx], dist_intrs[c_idx], dists[c_idx], image)
                            c_idx += 1
                        images.append(image)

                param_right = select_nf(params_right, nf)
                param_left = select_nf(params_left, nf)
                
                if abs_idx % args.stride == 0:
                    # visualizing the model
                    if args.vis_smpl:
                        vertices_right = body_model_right(return_verts=True, return_tensor=False, **param_right)
                        vertices_left = body_model_left(return_verts=True, return_tensor=False, **param_left)
                        vertices = np.concatenate((vertices_left[0], vertices_right[0]), axis=0)
                        faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left[0].shape[0]), axis=0)
                        image_vis = vis_smpl(args, vertices=vertices, faces=faces, images=images, nf=nf, cameras=cameras, add_back=True, out_dir=outhand_mano_path)
                        if abs_idx == 0:
                            outhand_mano = create_video_writer(outhand_mano_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_mano.write(image_vis)

                    # save the mesh
                    if args.save_mesh:
                        vertices = np.concatenate((vertices_left[0], vertices_right[0]), axis=0)
                        faces = np.concatenate((body_model_left.faces, body_model_right.faces+vertices_left[0].shape[0]), axis=0)
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        outdir = os.path.join(output_path, f'meshes/{str(selected_vid_idx).zfill(3)}')
                        os.makedirs(outdir, exist_ok=True)
                        outname = os.path.join(outdir, '{:08d}.obj'.format(nf))
                        mesh.export(outname)
                        
                    # visualize keypoint reprojection
                    vis_config = CONFIG['handlr']
                    if args.vis_2d_repro:
                        keypoints2d = np.concatenate((all_keypoints2d_right[abs_idx], all_keypoints2d_left[abs_idx]), axis=1)
                        kpts_repro = keypoints2d
                        image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_2d_path)
                        if abs_idx == 0:
                            outhand_2d = create_video_writer(outhand_2d_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_2d.write(image_vis)
                        
                    if args.vis_3d_repro:
                    #     keypoints = body_model(return_verts=False, return_tensor=False, **param)[0]
                        keypoints = np.concatenate((keypoints3d_right[abs_idx], keypoints3d_left[abs_idx]), axis=0)
                        kpts_repro = projectN3(keypoints, projs)
                        kpts_repro[:, :, 2] = 0.5
                        image_vis = vis_repro(args, images, kpts_repro, config=vis_config, nf=nf, mode='repro_smpl', outdir=outhand_3d_path)
                        if abs_idx == 0:
                            outhand_3d = create_video_writer(outhand_3d_path, (image_vis.shape[1], image_vis.shape[0]), fps=30)
                        outhand_3d.write(image_vis)
                    
                nf += 1

            if not args.save_frame:
                if args.vis_smpl:
                    outhand_mano.release()
                    convert_video_ffmpeg(outhand_mano_path)
                    print('Video Handler Released')
                if args.vis_2d_repro:
                    outhand_2d.release()
                    convert_video_ffmpeg(outhand_2d_path)
                    print('Video Handler Released')
                if args.vis_3d_repro:
                    outhand_3d.release()
                    convert_video_ffmpeg(outhand_3d_path)
                    print('Video Handler Released')
