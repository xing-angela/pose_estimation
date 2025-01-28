import os
import sys
import torch
import argparse

sys.path.append(".")
import src.utils.params as param_utils
from src.utils.reader_v2 import Reader
from src.utils.instantngp_helper import *
from src.utils.cameras import removed_cameras, map_camera_names, get_projections, get_ngp_cameras
from scipy.spatial.transform import Rotation

sys.path.append("/instant-ngp/build")
import pyngp as ngp

def ngp_train(args, params, cam_names, cam_mapper, all_view_images, all_view_masks, intrs, extrs, dists, frame_id, mesh_dir):
    print("creating testbed")
    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    testbed.reload_network_from_file(args.network)
    print("created testbed")
    
    ## Remove camera from params
    params = [param for param in params if (param["cam_name"] in cam_names) and (param["cam_name"] in cam_mapper)][::1]
    params = np.asarray(params)
    
    testbed.create_empty_nerf_dataset(
        n_images=len(params), aabb_scale=args.aabb_scale
    )
    print(f"Training on {len(params)} views.")
    id_ = 0
    
    imgs_full = []
    imgs = []
    img_names = []

    for idx, param in enumerate(params):
        if param["cam_name"] not in all_view_masks.keys():
            continue
        # if (param["cam_name"] not in cam_names) and (param["cam_name"] not in cam_mapper):
        #     continue
        img_cv2 = all_view_images[param['cam_name']][frame_id]
        mask = all_view_masks[param['cam_name']][frame_id]

        if mask is not None:
            combined_mask = np.any(mask, axis=0)  # Shape: [h, w]
            alpha_channel = np.where(combined_mask, 255, 0).astype(np.uint8)  # Opaque (255) or Transparent (0)
            rgba_img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGBA)
            rgba_img[:, :, 3] = alpha_channel

            img = rgba_img.astype(np.float32)
            img /= 255

            imgs.append(img)
            img_full = img.copy()
            img_full[:, :, 3] = np.ones_like(np.array(img_full[:, :, 3]))
            imgs_full.append(img_full)
            img_names.append(param['cam_name'])
            depth_img = np.zeros((img.shape[0], img.shape[1]))
            img = srgb_to_linear(img)
            # # premultiply
            img[..., :3] *= img[..., 3:4]

            extr = extrs[idx]
            intr = intrs[idx]
            dist = dists[idx]

            testbed.nerf.training.set_image(id_, img, depth_img)
            testbed.nerf.training.set_camera_extrinsics(id_, extr[:3], convert_to_ngp=False)
            testbed.nerf.training.set_camera_intrinsics(
                id_,
                fx=param["fx"],
                fy=param["fy"],
                cx=param["cx"],
                cy=param["cy"],
                k1=param["k1"],
                k2=param["k2"],
                p1=param["p1"],
                p2=param["p2"],
            )
            id_ += 1

    # Taken from i-ngp:scripts/run.py
    # testbed.color_space = ngp.ColorSpace.SRGB
    testbed.nerf.visualize_cameras = True
    testbed.background_color = [0.0, 0.0, 0.0, 0.0]
    testbed.nerf.training.random_bg_color = True
    testbed.training_batch_size = args.batch_size

    testbed.nerf.training.n_images_for_training = id_

    testbed.shall_train = True
    testbed.nerf.training.optimize_extrinsics = False
    testbed.nerf.training.optimize_focal_length = args.optimize_focal_length
    testbed.nerf.training.optimize_distortion = args.optimize_distortion
    testbed.nerf.cone_angle_constant = 0.000

    n_steps = args.n_steps
    old_training_step = 0
    tqdm_last_update = 0

    start = time.time()
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():
                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break

                # if testbed.training_step == n_steps // 2:
                #     for idx in range(id_):
                #         img_full = imgs_full[idx]
                #         img_full = srgb_to_linear(img_full)
                #         # # premultiply
                #         img_full[..., :3] *= img_full[..., 3:4]
                #         testbed.nerf.training.set_image(idx, img_full, depth_img)

                # Update progress bar
                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    testbed.shall_train = False
    testbed.nerf.cone_angle_constant = 0.0
    end = time.time()

    # seg_dir = os.path.join(args.out_dir, "gt_contacts", f"{args.handedness}_hand", "data", args.save_dir_name)
    # os.makedirs(seg_dir, exist_ok=True)
    # save_mask(testbed, imgs, img_names, params, seg_dir, args.num_objects)

    mesh_path = os.path.join(mesh_dir, "mesh.ply")
    save_mesh(
        args.marching_cubes_res, 
        mesh_path, testbed, 
        args.downscale_factor, 
        pad=args.pad, 
        num_objects = 2, #args.num_objects, 
        aabb_mn = None,
        aabb_mx = None,
        refine = True
    )

    # fields = params.dtype.fields
    # if args.optimize_extrinsics:
    #     print("saving optimized extrinsics")
    #     optim_params = params.copy()
    #     for idx, param in enumerate(params):
    #         c2w = testbed.nerf.training.get_camera_extrinsics(idx, convert_to_ngp=False)
    #         c2w = np.vstack((c2w, np.asarray([[0, 0, 0, 1]])))
    #         w2c = np.linalg.inv(c2w)
    #         qvec = Rotation.from_matrix(w2c[:3, :3]).as_quat()
    #         tvec = w2c[:3, 3]
    #         optim_params[idx]["qvecx"] = qvec[0]
    #         optim_params[idx]["qvecy"] = qvec[1]
    #         optim_params[idx]["qvecz"] = qvec[2]
    #         optim_params[idx]["qvecw"] = qvec[3]
    #         optim_params[idx]["tvecx"] = tvec[0]
    #         optim_params[idx]["tvecy"] = tvec[1]
    #         optim_params[idx]["tvecz"] = tvec[2]

    #     np.savetxt(
    #         os.path.join(mesh_dir, "optim_params.txt"),
    #         optim_params,
    #         fmt="%s",
    #         header=" ".join(fields),
    #     )
    
    # success, vertices = save_raw_density(testbed, 128, mesh_dir, frame_id)
    # return success, vertices


def main():
    parser = argparse.ArgumentParser(description="Instant NGP Mesh")
    parser.add_argument("--ith", type=int, default=14)
    parser.add_argument("--root_dir", "-r", type=str)
    parser.add_argument("--out_dir", "-o", type=str)
    parser.add_argument("--anchor_camera", type=str, default="brics-odroid-001_cam0")
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--session", type=str, default='2025-01-08')
    parser.add_argument("--text_prompt", type=str, default="book")
    parser.add_argument("--use_optim_params", action="store_true")
    parser.add_argument("--remove_bottom", action="store_true", default=False)
    parser.add_argument("--remove_side", action="store_true", default=True)
    parser.add_argument("--input_type", type=str, default="video", choices=["video", "image"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--handedness", choices=["left", "right"], default="right", type=str)
    add_ngp_parser(parser)
    args = parser.parse_args()

    save_main_path = os.path.join(args.out_dir, "object_tracking")
    # save_main_path = os.path.join(args.out_dir, "gt_contacts", f"{args.handedness}_hand")
    if args.input_type == "video":
        input_path = os.path.join(args.root_dir, args.session)
        # save_segmentation_path = os.path.join(save_main_path, "segmentation_"+args.text_prompt)
        save_segmentation_path = os.path.join(args.out_dir, "data", "mask")
        # save_segmentation_path = os.path.join(save_main_path, "data", "segmented_sam2")
    else:
        # save_segmentation_path = os.path.join(args.root_dir, args.session, "data", "segmented_ngp")
        save_segmentation_path = os.path.join(args.out_dir, "data", "mask")
        input_path = os.path.join(args.root_dir, args.session, "data", "segmented_ngp")
        # save_segmentation_path = os.path.join(save_main_path, "segmentation_"+args.text_prompt)
    
    mesh_dir = os.path.join(save_main_path, "mesh", "ngp_mesh")

    # Prepare for cameras and video readers
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
    reader = Reader(args.input_type, input_path, undistort=args.undistort, cam_path=params_path, cams_to_remove=cams_to_remove, ith=args.ith, start_frame=args.start, end_frame=args.end, anchor_camera=args.anchor_camera, extn="jpg")

    # Loading the multiview segmentation masks
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
            for f_idx in range(reader.frame_count)[0::args.gap*10]:
                f_idx = 30
                if args.input_type == "video":
                    img_name = str(f_idx).zfill(4) + '.png'
                    # img_name = str(f_idx).zfill(6) + '.png'
                else:
                    img_name = str(f_idx).zfill(8) + '.png'
                    # img_name = str(f_idx).zfill(6) + '.png'
                img_path = os.path.join(save_segmentation_path, camera_name, img_name)
                image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
                image = cv2.undistort(image, intrs[v_idx], dists[v_idx], None, dist_intrs[v_idx])
                orig_imgs.append(image[:, :, :3][:, :, ::-1])
                all_masks.append(image[:, :, 3][np.newaxis, :, :] > 0)
                break

            all_view_images[camera_name] = orig_imgs
            all_view_masks[camera_name] = all_masks
        else:
            cam_no_masks.append(camera_name)
    
    for cam in cam_no_masks:
        if cam in cam_names:
            cam_names.remove(cam)

    if args.align_bounding_box:
        with open(args.cam_faces_path, "r") as f:
            faces = json.load(f)
    else:
        faces = None

    
    intrs, extrs, dists = get_ngp_cameras(args, params, cam_names, cam_mapper, align_bounding_box=args.align_bounding_box, faces=faces)

    translation_success = False
    # sorted_frame_id_list = sort_frames_by_mask_area(all_view_masks, num_frames=len(all_masks))
    # print(sorted_frame_id_list)
    # exit()
    sorted_frame_id_list = [0]

    for abs_idx, anchor_frame_id in enumerate(sorted_frame_id_list):
        print(f'The best frame id is {anchor_frame_id}')
        translation_success, vertices = ngp_train(args, params, cam_names, cam_mapper, all_view_images, all_view_masks, intrs, extrs, dists, anchor_frame_id, mesh_dir=mesh_dir)
        if translation_success:
            lower_bound = np.percentile(vertices, 5, axis=0)
            upper_bound = np.percentile(vertices, 95, axis=0)
            center_of_boundary = (lower_bound + upper_bound) / 2
            vertices = center_of_boundary[np.newaxis, :].astype(np.float32)
            break


if __name__ == "__main__":
    main()