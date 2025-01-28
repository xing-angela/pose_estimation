import os
import sys
import cv2
import json
import torch
import argparse
import open_clip
import numpy as np

from PIL import Image
from tqdm import tqdm

sys.path.append(".")
import src.utils.params as param_utils
from src.utils.reader_v2 import Reader
import src.utils.colmap_utils as colmap_utils
from src.utils.video_handler import create_video_writer, convert_video_ffmpeg
from src.utils.cameras import removed_cameras, map_camera_names, get_projections
from src.utils.template_util import images_to_template_reps, match_image_with_template_reps
from src.utils.pytorch3d_utils import batch_render_loader, batch_render_ref_loader, DRModel, check_for_nan_params, visualize_image_list, alpha_blend, save_transformed_mesh, uniform_quaternions

sys.path.append("./submodules/dinov2")
import src.utils.dinov2_utils as dinov2_utils

# import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes, load_ply
from pytorch3d.transforms import Rotate, Translate, quaternion_to_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
open_clip_model.eval()

def main():
    parser = argparse.ArgumentParser(description="Object Tracking")
    parser.add_argument("--ith", type=int, default=14)
    parser.add_argument("--root_dir", "-r", type=str)
    parser.add_argument("--out_dir", "-o", type=str)
    parser.add_argument("--scan_path", type=str, default='book_simplified.obj')
    parser.add_argument("--anchor_camera", type=str, default="brics-odroid-001_cam0")
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--session", type=str, default='2025-01-08')
    parser.add_argument("--text_prompt", type=str, default="book")
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument("--remove_bottom", action="store_true", default=False)
    parser.add_argument("--remove_side", action="store_true", default=True)
    parser.add_argument("--input_type", type=str, default="video", choices=["video", "image"])
    parser.add_argument("--to_filter_views", action="store_true")
    parser.add_argument('--down', default=4, type=float, help='downsample image size')
    parser.add_argument('--camera_type', default='perspective', help='')
    parser.add_argument('--texture_type', default='ply', help='single|multi|ply')
    parser.add_argument('--shader', default='phong', help='phong|mask')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--normalize_verts', default=False, action='store_true')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    if args.input_type == "video":
        input_path = os.path.join(args.root_dir, args.session)
    else:
        input_path = os.path.join(args.root_dir, args.session, "data", "image")
    save_main_path = os.path.join(args.out_dir, "object_tracking")
    save_segmentation_path = os.path.join(save_main_path, "segmentation_"+args.text_prompt)
    mesh_dir = os.path.join(save_main_path, "mesh", "ngp_mesh")
    scanned_mesh_path = os.path.join(args.out_dir, "scan", args.scan_path)

    positive_text_prompts = [args.text_prompt]
    negative_text_prompts = ["hands"]

    if args.use_optim_params:
        params_txt = "optim_params.txt"
    else:
        params_txt = "params.txt"

    # Prepare for cameras and video readers
    params_path = os.path.join(args.out_dir, params_txt)
    params = param_utils.read_params(params_path)
    cam_names = list(params[:]["cam_name"])
    removed_camera_path = os.path.join(args.out_dir, 'ignore_camera.txt')
    if os.path.isfile(removed_camera_path):
        with open(removed_camera_path) as file:
            ignored_cameras = [line.rstrip() for line in file]
    else:
        ignored_cameras = None
    cams_to_remove = removed_cameras(args.input_type, remove_side=args.remove_side, remove_bottom=args.remove_bottom, ignored_cameras=ignored_cameras)

    for cam in cams_to_remove:
        if cam in cam_names:
            cam_names.remove(cam)
    cam_mapper = map_camera_names(input_path, cam_names)

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
            if folder not in cam_names:
                print(folder)
                cams_to_remove.append(folder)

    intrs, projs, dist_intrs, dists, cameras = get_projections(args, params, cam_names, cam_mapper, easymocap_format=True)
    reader = Reader(args.input_type, input_path, undistort=True, cam_path=params_path, cams_to_remove=cams_to_remove, ith=args.ith, start_frame=args.start, end_frame=args.end, anchor_camera=args.anchor_camera)

    # print(render_cameras[:])

    tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
    tokenized_text = tokenizer(positive_text_prompts + negative_text_prompts)
    text_features = open_clip_model.encode_text(tokenized_text)

    # Load multiview segmentation masks
    cam_no_masks = []
    all_view_masks = {}
    all_view_images = {}
    for v_idx, input_video_path in tqdm(enumerate(reader.views), total=len(reader.views), desc="Loading segmentations across views"):
        if args.input_type == "video":
            camera_name = input_video_path.split('/')[-1].rpartition('_')[0]
        else:
            camera_name = input_video_path
        if os.path.exists(os.path.join(save_segmentation_path, camera_name)):
            orig_imgs = []
            all_masks = []
            num_frames = 0
            for f_idx in range(reader.frame_count)[0::args.gap]:
                img_path = os.path.join(save_segmentation_path, camera_name, str(f_idx).zfill(6) + '.png')
                orig_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                image = cv2.undistort(orig_image, intrs[v_idx], dists[v_idx], None, dist_intrs[v_idx])
                if args.to_filter_views and f_idx == 0:
                    is_to_filter = False
                    bgr_image = image[:, :, :3].copy()
                    rgb_image = bgr_image[:, :, ::-1]
                    mask = image[:, :, 3] > 0
                    rgb_image[~mask] = 0
                    preprocess_image = open_clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0)
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        image_features = open_clip_model.encode_image(preprocess_image)
                        text_features = open_clip_model.encode_text(tokenized_text)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        max_prob, max_id = torch.max(text_probs, dim=-1)
                        if max_id.item() == 0:
                            orig_imgs.append(image[:, :, :3][:, :, ::-1])
                            all_masks.append(image[:, :, 3][np.newaxis, :, :] > 0)
                        else:
                            is_to_filter = True
                else:
                    orig_imgs.append(image[:, :, :3][:, :, ::-1])
                    all_masks.append(image[:, :, 3][np.newaxis, :, :] > 0)

                num_frames += 1
            
            if args.to_filter_views and is_to_filter:
                print(f"Filtered {camera_name}")
                cam_no_masks.append(camera_name)
            else:
                all_view_images[camera_name] = orig_imgs
                all_view_masks[camera_name] = all_masks
        else:
            cam_no_masks.append(camera_name)

    for cam in cam_no_masks:
        if cam in cam_names:
            cam_names.remove(cam)

    random_ranges = torch.tensor([0.1, 0.0, 0.1])
    anchor_frame_id = 0
    translation_success = False
    mesh_path= os.path.join(mesh_dir, 'volume_raw', f"{str(0).zfill(6)}_filtered_bounded_clutered_samples.ply")
    vertices, faces = load_ply(mesh_path)
    vertices = vertices.cpu().numpy()
    lower_bound = np.percentile(vertices, 5, axis=0)
    upper_bound = np.percentile(vertices, 95, axis=0)
    center_of_boundary = (lower_bound + upper_bound) / 2
    vertices = center_of_boundary[np.newaxis, :].astype(np.float32)

    save_render_folder = os.path.join(save_main_path, 'render')
    os.makedirs(save_render_folder, exist_ok = True)
    save_pose_folder = os.path.join(save_main_path, 'pose')
    os.makedirs(save_pose_folder, exist_ok = True)
    save_transform_mesh_path = os.path.join(save_pose_folder, 'transform_mesh.obj')

    # Step 3: Multi-view optimize object pose
    extractor = dinov2_utils.DinoFeatureExtractor(model_name="dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1").to(device)
    
    """ initialize T from reconstruction"""
    print(vertices)
    mesh_translation = torch.Tensor(np.mean(vertices, axis=0)[np.newaxis, :].astype(np.float32))
    print(mesh_translation)

    """ load camera """
    render_cameras = colmap_utils.read_cameras_from_txt(params_path, cam_names, cam_mapper)

    """ load scanned mesh """
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        template_mesh = load_objs_as_meshes([scanned_mesh_path], device=device)

    num_retrieval_initializations = 100
    retrieval_rotation_matrices = uniform_quaternions(num_retrieval_initializations)
    # retrieval_rotation_matrices = quaternion_to_axis_angle(random_quats)
    default_translation = [0.0, 0.0, 0.0]

    """ Round0: initialize with NGP mean; Round1: initalize with center. """
    for init_round in range(2):
        """ load renders """
        renderer_list = batch_render_loader(args, render_cameras, device)
        
        if init_round == 1:
            translation_success = False
            mesh_translation = torch.Tensor(np.mean(default_translation, axis=0)[np.newaxis, :].astype(np.float32))
        
        templates_list = []
        """ initialize random views """
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            for mid, mesh_rotation in tqdm(enumerate(retrieval_rotation_matrices)):
                # Render the image using the updated camera position. Based on the new position of the 
                # camera we calculate the rotation and translation matrices
                R = Rotate(quaternion_to_matrix(mesh_rotation.to(device)), orthogonal_tol=1e-3)
                T = Translate(torch.clamp(mesh_translation.to(device), min=-0.2, max=1.2))   # (1, 3)
                transform  = R.compose(T)
                tverts = transform.transform_points(template_mesh.verts_list()[0])
                faces = template_mesh.faces_list()[0]
                tmesh = Meshes(
                    verts=[tverts],   
                    faces=[faces],
                    textures = template_mesh.textures,
                )
                
                template_info = {'mesh_rotation': mesh_rotation.cpu().numpy().tolist()}
                for rid, render_info in enumerate(renderer_list):
                    cam_name =render_info['cam_name']
                    image_tensor, fragments = render_info['renderer'](meshes_world=tmesh)
                    image_np = (image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    template_info[cam_name] = image_np
                templates_list.append(template_info)

        """ Retrieval Comparison Across Views """
        all_init_scores = [0] * num_retrieval_initializations
        all_init_freq = [0] * num_retrieval_initializations
        all_init_dist = [0] * num_retrieval_initializations
        best_view_score = []
        for camera in render_cameras:
            anchor_cam_name = camera['cam_name']
            template_imgs = []
            for template_info in templates_list:
                template_imgs.append(template_info[anchor_cam_name])

            valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs = images_to_template_reps(template_imgs, extractor, device)
            if valid_template_ids is None:
                continue

            instance_image_rgb = all_view_images[anchor_cam_name][anchor_frame_id]
            instance_image_mask = all_view_masks[anchor_cam_name][anchor_frame_id]
            if instance_image_rgb is None or instance_image_mask is None:
                continue
            combined_mask = np.any(instance_image_mask, axis=0)  # Shape: [h, w]
            alpha_channel = np.where(combined_mask, 255, 0).astype(np.uint8)  # Opaque (255) or Transparent (0)
            instance_image = cv2.cvtColor(instance_image_rgb, cv2.COLOR_RGB2RGBA)
            instance_image[:, :, 3] = alpha_channel
            
            template_scores, sorted_template_ids, valid_template_ids = match_image_with_template_reps(instance_image, extractor, valid_template_ids, feat_raw_projectors, feat_cluster_centroids, template_descs, feat_cluster_idfs, device)
            for s, sid in zip(template_scores, sorted_template_ids):
                if s > 0.2:
                    tid = valid_template_ids[sid]
                    all_init_scores[tid] += s
                    all_init_freq[tid] += 1


        all_init_scores = torch.tensor(all_init_scores)
        sorted_indices = torch.argsort(all_init_scores, descending=True)
        sorted_scores = all_init_scores[sorted_indices]


        # Number of initializations
        num_initializations = 15 if init_round == 1 else 10
        init_optimization_its = 1000 if init_round == 1 else 600
        final_optimization_its = 1000 if init_round == 1 else 600
        refine_optimization_its = 400
        init_lr = 2e-3
        final_lr = 1e-3
        refine_lr = 1e-3
        lambda_rgb = 0.0

        print("Sorted scores:", sorted_scores[:num_initializations])
        rotation_matrices = retrieval_rotation_matrices[sorted_indices[:num_initializations]]
        """ Optimize First Frame  """
        # Load the first frame
        image_ref_list, renderer_list, n_mask_pixel = batch_render_ref_loader(args, anchor_frame_id, render_cameras, all_view_images, all_view_masks, device)

        """ Find best R"""
        # Store best loss and best model state
        best_loss = float('inf')
        best_init_R = None
        best_model_state_dict = None
        best_it = 0
        best_iou = 0
        converge_loss = n_mask_pixel // 5000
        all_iou = np.zeros((num_initializations,))

        # Loop over all num_initializations
        for init_idx, init_R in enumerate(rotation_matrices):
            print(f"Optimizing initialization {init_idx} / {num_initializations}")
            opt_filename_output = f"{save_render_folder}/optimization_demo_r{init_idx}.mp4"

            # Initialize the model with the current anchor_T
            random_offset = (torch.rand_like(mesh_translation) * 2 - 1) * random_ranges
            model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=mesh_translation if translation_success else mesh_translation + random_offset, init_R=init_R, lambda_rgb=lambda_rgb).to(device)
            optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=init_lr)

            # Training loop for 100 epochs
            loop = tqdm(range(init_optimization_its))
            for i in loop:
                with torch.autocast(device_type="cuda", dtype=torch.float32):
                    optimizer.zero_grad()
                    loss_pixels_list, image_list, _ = model(i)

                    loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                    loss.backward()
                    # Check gradients before clipping
                    has_grad_nan = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:  # Ensure the parameter has a gradient
                            grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                            if torch.isnan(param.grad).any():
                                has_grad_nan = True
                                print(f"NaN detected in gradient for parameter: {name}")
                    if has_grad_nan:
                        break
                    # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                    optimizer.step()
                    model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)

                    loop.set_description(f'Init {init_idx}: Optimizing (loss {loss.item():.4f})')

                    # Save outputs to create a GIF. 
                    if i % 10 == 0:
                        render_image = visualize_image_list(image_list, save_alpha=True)
                        ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                        image = alpha_blend(render_image, ref_image)[..., :3]
                    
                    if i == 0:
                        writer = create_video_writer(opt_filename_output, (image.shape[1], image.shape[0]), fps=init_optimization_its//4)
                    writer.write(image)

                    if loss.item() < converge_loss:
                        break

            writer.release()
            convert_video_ffmpeg(opt_filename_output)

            """ Calculate IOU"""
            all_intersection = 0
            all_union = 0
            for img, ref in zip(image_list, image_ref_list):
                img_mask = (img[0, :, :, 3] > 0).detach().squeeze().cpu().numpy()
                ref_mask = ref[:, :, 3]
                all_intersection += np.logical_and(img_mask, ref_mask).sum()
                all_union += np.logical_or(img_mask, ref_mask).sum()

            iou = all_intersection / all_union if all_union != 0 else 0
            print(f"Init {init_idx}: IOU {iou:.4f}")
            all_iou[init_idx] = iou

            # Check if the current initialization yields a lower loss
            if loss.item() < best_loss and not check_for_nan_params(model):
                best_it = init_idx
                best_loss = loss.item()
                best_init_R = init_R
                best_model_state_dict = model.state_dict()  # Save the model's state dict for later use
                best_iou = iou


        # After the initial optimization for all initializations, we continue with the best one
        print(f"Best initialization found with loss {best_loss:.4f} at iteration {best_it}")
        
        if np.any(all_iou > 0.45):
            break
        if np.all(all_iou > 0.10):
            break

    """ Optimize the anchor frame"""
    opt_filename_output = f"{save_render_folder}/optimization_demo_best.mp4"

    model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=mesh_translation, init_R=best_init_R, lambda_rgb=lambda_rgb).to(device)
    model.load_state_dict(best_model_state_dict)
    optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=final_lr)

    # Store best loss and best model state
    best_loss = float('inf')
    best_model_state_dict = None
    best_it = 0
    converge_loss = n_mask_pixel // 500

    loop = tqdm(range(final_optimization_its))
    for i in loop:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            optimizer.zero_grad()
            loss_pixels_list, image_list, tmesh = model(i)

            loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
            loss.backward()
            # Check gradients before clipping
            has_grad_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:  # Ensure the parameter has a gradient
                    grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                    if torch.isnan(param.grad).any():
                        has_grad_nan = True
                        print(f"NaN detected in gradient for parameter: {name}")
            if has_grad_nan:
                break
            # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
            optimizer.step()
            model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)
            
            loop.set_description(f'Refine Initialization Optimizing ({loss.data:.4f})' )

            # Check if the current initialization yields a lower loss
            if loss.item() < best_loss:
                best_it = i
                best_loss = loss.item()
                best_model_state_dict = model.state_dict()  # Save the model's state dict for later use

            # Save outputs to create a GIF. 
            if i % 10 == 0:
                render_image = visualize_image_list(image_list, save_alpha=True)
                ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                image = alpha_blend(render_image, ref_image)[..., :3]

            if i == 0:
                writer = create_video_writer(opt_filename_output, (image.shape[1], image.shape[0]), fps=final_optimization_its//4)
            writer.write(image)

            if loss.item() < converge_loss:
                break

    writer.release()
    convert_video_ffmpeg(opt_filename_output)
    save_transformed_mesh(tmesh, template_mesh, save_transform_mesh_path)

    """ Optimize all frames"""
    """ find total frames"""

    opt_filename_output = f"{save_render_folder}/optimization_demo_all.mp4"
    video_writer_dict = {}
    pose_writer = f"{save_pose_folder}/optimized_pose.json"
    pose_dict = {}
    all_init_state_dict = best_model_state_dict.copy()

    ## >>Forward pose optimization
    # for frame_id in range(anchor_frame_id, reader.frame_count)[0::args.gap]:
    for i in range(num_frames):
        frame_id = i * args.gap
            
        image_ref_list, renderer_list, n_mask_pixel = batch_render_ref_loader(args, i, render_cameras, all_view_images, all_view_masks, device)

        model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=best_model_state_dict['mesh_translation'], init_R=best_model_state_dict['mesh_rotation'], lambda_rgb=lambda_rgb).to(device)
        # model.load_state_dict(best_model_state_dict)
        model_init_state = model.state_dict()
        optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)
        
        # Store best loss and best model state
        best_loss = float('inf')
        # best_model_state_dict = None
        best_it = 0
        
        converge_loss = n_mask_pixel // 500
        loop = tqdm(range(refine_optimization_its))
        for i in loop:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                optimizer.zero_grad()
                loss_pixels_list, image_list, tmesh = model(i)

                loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                loss.backward()
                # Check gradients before clipping
                has_grad_nan = False
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Ensure the parameter has a gradient
                        grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                        if torch.isnan(param.grad).any():
                            has_grad_nan = True
                            print(f"NaN detected in gradient for parameter: {name}")
                if has_grad_nan:
                    break
                # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                optimizer.step()
                model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)
                
                loop.set_description(f'Frame {frame_id}/{reader.frame_count} Optimizing {loss.item():.4f}')

                has_nan = check_for_nan_params(model)
                if has_nan:
                    model.load_state_dict(all_init_state_dict)
                    optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)

                # Check if the current initialization yields a lower loss
                if loss.item() < best_loss and not has_nan:
                    best_it = i
                    best_loss = loss.item()
                    best_model_state_dict = model.state_dict()  # Save the model's state dict for later use

                # Save outputs to create a GIF. 
                if i == refine_optimization_its - 1 or loss.item() < converge_loss and not has_nan:
                    render_image = visualize_image_list(image_list, save_alpha=True)
                    ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                    image = alpha_blend(render_image, ref_image)[..., :3]
                    
                    
                    video_writer_dict[str(frame_id).zfill(6)] = image
                    mesh_translation_list = best_model_state_dict['mesh_translation'].cpu().numpy().tolist()  # Convert to NumPy, then to list
                    mesh_rotation_list = best_model_state_dict['mesh_rotation'].cpu().numpy().tolist()        # Convert to NumPy, then to list

                    pose_dict[str(frame_id).zfill(6)] = {
                        'mesh_translation': mesh_translation_list,
                        'mesh_rotation': mesh_rotation_list
                    }

                if loss.item() < converge_loss and not has_nan:
                    break

    best_model_state_dict = all_init_state_dict.copy()
    ## >>backward pose optimization
    # for frame_id in range(0, anchor_frame_id)[0::args.gap][::-1]:
    for i in range(num_frames)[::-1]:
        frame_id = i * args.gap
            
        image_ref_list, renderer_list, n_mask_pixel = batch_render_ref_loader(args, i, render_cameras, all_view_images, all_view_masks, device)

        model = DRModel(meshes=template_mesh, renderer_list=renderer_list, image_ref_list=image_ref_list, anchor_T=best_model_state_dict['mesh_translation'], init_R=best_model_state_dict['mesh_rotation'], lambda_rgb=lambda_rgb).to(device)
        # model.load_state_dict(best_model_state_dict)
        model_init_state = model.state_dict()
        optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)
        
        # Store best loss and best model state
        best_loss = float('inf')
        # best_model_state_dict = None
        best_it = 0
        
        converge_loss = n_mask_pixel // 500
        loop = tqdm(range(refine_optimization_its))
        for i in loop:
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                optimizer.zero_grad()
                loss_pixels_list, image_list, tmesh = model(i)

                loss = torch.sum(torch.stack(loss_pixels_list)) / len(loss_pixels_list)
                loss.backward()
                # Check gradients before clipping
                has_grad_nan = False
                for name, param in model.named_parameters():
                    if param.grad is not None:  # Ensure the parameter has a gradient
                        grad_norm = param.grad.norm().item()  # Calculate the norm of the gradient
                        if torch.isnan(param.grad).any():
                            has_grad_nan = True
                            print(f"NaN detected in gradient for parameter: {name}")
                if has_grad_nan:
                    break
                # torch.nn.utils.clip_grad_norm_([model.mesh_translation, model.mesh_rotation], max_norm=1.0)
                optimizer.step()
                model.mesh_rotation.data = torch.nn.functional.normalize(model.mesh_rotation.data, dim=0)
                
                loop.set_description(f'Frame {frame_id}/{reader.frame_count} Optimizing {loss.item():.4f}')

                has_nan = check_for_nan_params(model)
                if has_nan:
                    model.load_state_dict(model_init_state)
                    optimizer = torch.optim.Adam([model.mesh_rotation, model.mesh_translation], lr=refine_lr)

                # Check if the current initialization yields a lower loss
                if loss.item() < best_loss and not has_nan:
                    best_it = i
                    best_loss = loss.item()
                    best_model_state_dict = model.state_dict()  # Save the model's state dict for later use

                # Save outputs to create a GIF. 
                if i == refine_optimization_its - 1 or loss.item() < converge_loss and not has_nan:
                    render_image = visualize_image_list(image_list, save_alpha=True)
                    ref_image = visualize_image_list(image_ref_list, save_alpha=True)
                    image = alpha_blend(render_image, ref_image)[..., :3]
                    
                    
                    video_writer_dict[str(frame_id).zfill(6)] = image
                    mesh_translation_list = best_model_state_dict['mesh_translation'].cpu().numpy().tolist()  # Convert to NumPy, then to list
                    mesh_rotation_list = best_model_state_dict['mesh_rotation'].cpu().numpy().tolist()        # Convert to NumPy, then to list

                    pose_dict[str(frame_id).zfill(6)] = {
                        'mesh_translation': mesh_translation_list,
                        'mesh_rotation': mesh_rotation_list
                    }

                if loss.item() < converge_loss and not has_nan:
                    break

    with open(pose_writer, 'w') as f:
        json.dump(pose_dict, f, indent=4)  # indent=4 makes the JSON file more readable

    sorted_video_writer_dict = dict(sorted(video_writer_dict.items(), key=lambda x: int(x[0])))
    writer = create_video_writer(opt_filename_output, (image.shape[1], image.shape[0]), fps=10)
    for fid, image in sorted_video_writer_dict.items():
        writer.write(image)
    writer.release()
    convert_video_ffmpeg(opt_filename_output)

    print(f"Data saved to {pose_writer}")




if __name__ == "__main__":
    main()