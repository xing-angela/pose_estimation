import os
import gc
import cv2
import sys
import torch
import argparse
import open_clip
import numpy as np
import src.utils.params as param_utils

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.append(".")
from src.utils.reader_v2 import Reader
from src.utils.video_handler import frame_preprocess
from src.utils.cameras import removed_cameras, map_camera_names, get_projections

sys.path.append("./Grounded-SAM-2")
import supervision as sv
from src.sam2_video_predictor import build_sam2_video_predictor

device = "cuda" if torch.cuda.is_available() else "cpu"

def track_and_save_video(args, grounding_processor, grounding_model, video_predictor, text_prompt, text_features, text_num, images_cv2, open_clip_model, open_clip_preprocess, save_folder='test.png', view_name='', save_video=True, save_frame=True):
    # prompt grounding dino to get the box coordinates on specific frame
    gap = args.gap
    total_length = len(images_cv2)
    interval = max(total_length // 2 * gap, 1)
    sampled_frame_indices = list(range(0, total_length, interval))
    # sampled_frame_indices = [0]
    
    all_results = []
    for ann_frame_idx in sampled_frame_indices:
        image = Image.fromarray(cv2.cvtColor(images_cv2[ann_frame_idx], cv2.COLOR_BGR2RGB)).convert("RGB")

        # run Grounding DINO on the image
        inputs = grounding_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        all_results.append(results[0])

    # TODO: we assume we only have one candidate.
    all_best_ann_frame_idx = 0
    all_best_score = 0
    all_best_box = None
    all_best_label = ''
    for ann_frame_idx, results in zip(sampled_frame_indices, all_results):
        boxes = results["boxes"].cpu().numpy() 
        labels = results["labels"]
        scores = results["scores"].cpu().numpy()

        valid_boxes = []
        valid_scores = []
        valid_labels = []
        
        for box, label, score in zip(boxes, labels, scores):
            if label.strip():
                valid_boxes.append(box)
                valid_scores.append(score)
                valid_labels.append(label)
        
        valid_boxes = np.array(valid_boxes)
        valid_scores = np.array(valid_scores)
        valid_labels = np.array(valid_labels)
        
        if len(valid_scores) > 0:
            max_score_idx = np.argmax(valid_scores)
            best_box = valid_boxes[max_score_idx]
            best_label = valid_labels[max_score_idx]
            best_score = valid_scores[max_score_idx]
            
            if best_score > all_best_score:
                all_best_ann_frame_idx, all_best_score, all_best_box, all_best_label = ann_frame_idx, best_score, best_box, best_label

    if all_best_box is None:
        return None, [None] * total_length

    all_best_ann_frame_idx_gap = all_best_ann_frame_idx // gap
    input_boxes = [all_best_box]
    OBJECTS = [all_best_label]
    scores = [all_best_score]

    # Using box prompt
    # init video predictor state
    images_cv2_gap = images_cv2[0::gap]
    inference_state = video_predictor.init_state(images_cv=images_cv2_gap)
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=all_best_ann_frame_idx_gap,
            obj_id=object_id,
            box=box,
        )

        if args.to_filter_views:
            bgr_image = images_cv2_gap[all_best_ann_frame_idx_gap].copy()
            rgb_image = bgr_image[:, :, ::-1]
            masks_dict = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            masks = list(masks_dict.values())
            masks = np.concatenate(masks, axis=0)
            combined_mask = np.any(masks, axis=0)
            mask_expanded = combined_mask[:, :, np.newaxis]
            rgb_image[~mask_expanded.repeat(3, axis=2)] = 0
            preprocess_image = open_clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = open_clip_model.encode_image(preprocess_image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                max_prob, max_id = torch.max(text_probs, dim=-1)
                if max_id.item() >= text_num:
                    return None, [None] * total_length

    # Propagate the video predictor to get the segmentation results for each frame
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=False):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, reverse=True):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    video_segments = dict(sorted(video_segments.items(), key=lambda x: x[0]))

    all_masks = [None] * total_length
    for gap_frame_idx, segments in video_segments.items():
        # Get the object IDs and masks
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        all_masks[gap_frame_idx * gap] = masks

    # Assuming OBJECTS and video_segments are defined
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}

    if save_video or save_frame:
        # Initialize the VideoWriter object
        frame_size = (image.size[0], image.size[1]) # Set the frame size (adjust as needed based on your frames)
        fps = 30  # Set frames per second

        # Use the 'mp4v' codec for MP4 files
        os.makedirs(save_folder, exist_ok = True)
        video_path = os.path.join(save_folder, view_name + '.mp4')
        if save_frame:
            frame_folder = os.path.join(save_folder, view_name)
            os.makedirs(frame_folder, exist_ok = True)
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

        # Iterate through each frame
        for gap_frame_idx, segments in video_segments.items():
            # Read the image/frame
            img = images_cv2[gap_frame_idx * gap]
            
            # Resize image if the size doesn't match the video size (optional, for consistency)
            img = cv2.resize(img, frame_size)
            
            # Get the object IDs and masks
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)

            if save_frame and gap_frame_idx % 1 == 0:
                combined_mask = np.any(masks, axis=0)
                alpha_mask = (combined_mask * 255).astype(np.uint8)
                img_rgba = np.dstack((img, alpha_mask))

                output_filename = os.path.join(frame_folder, str(gap_frame_idx * gap).zfill(6) + '.png')
                cv2.imwrite(output_filename, img_rgba)

            # Create detections using supervision library
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
                mask=masks,  # (n, h, w)
                class_id=np.array(object_ids, dtype=np.int32),
            )
            
            # Annotate the image with bounding boxes, labels, and masks
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            if save_video:
                video_writer.write(annotated_frame)
        if save_video:
            video_writer.release()

    return all_best_ann_frame_idx * gap, all_masks

def main():
    parser = argparse.ArgumentParser(description="Object Segmentation")
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
    parser.add_argument("--save_seg_video", action="store_true")
    parser.add_argument("--save_seg_frame", action="store_true")
    parser.add_argument("--input_type", "-t", default="video", choices=["video", "image"], help="Whether the input is a video or set of images")
    parser.add_argument("--undistort", action="store_true")
    parser.add_argument("--to_filter_views", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--handedness", choices=["left", "right"], default="right", type=str)
    args = parser.parse_args()

    if args.input_type == "video":
        input_path = os.path.join(args.root_dir, args.session)
    else:
        input_path = os.path.join(args.root_dir, args.session, "data", "image")
    save_main_path = os.path.join(args.out_dir, "object_tracking")
    save_segmentation_path = os.path.join(save_main_path, "segmentation_" + args.text_prompt)
    # save_main_path = os.path.join(args.out_dir, "gt_contacts", f"{args.handedness}_hand")
    # save_segmentation_path = os.path.join(save_main_path, "data", "segmented_sam2")
    os.makedirs(save_segmentation_path, exist_ok=True)

    positive_text_prompts = [args.text_prompt]
    negative_text_prompts = ["hands"]

    # Load DINO and SAM2
    # init sam image predictor and video predictor model
    # sam2_checkpoint = "Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
    sam2_checkpoint = "./assets/sam2_ckpts/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-base"
    grounding_processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

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
    reader = Reader(args.input_type, input_path, undistort=args.undistort, cam_path=params_path, cams_to_remove=cams_to_remove, ith=args.ith, start_frame=args.start, end_frame=args.end, anchor_camera=args.anchor_camera)

    open_clip_model, _, open_clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    open_clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
    tokenized_text = tokenizer(positive_text_prompts + negative_text_prompts)
    text_features = open_clip_model.encode_text(tokenized_text)

    # Step 1: Track object in videos across views -> 51 x 1000 frames. 
    # sam_text_prompt = '. '.join(positive_text_prompts)
    sam_text_prompt = args.text_prompt
    all_best_frame_ids = []
    cam_no_masks = []
    all_view_masks = {}
    all_view_images = {}
    for v_idx, input_video_path in tqdm(enumerate(reader.views), total=len(reader.views), desc="Segmenting across views"):
        if args.input_type == "video":
            camera_name = input_video_path.split('/')[-1].rpartition('_')[0]
            im_names, orig_imgs, im_h, im_w = frame_preprocess(input_video_path, args.undistort, intrs[v_idx], dist_intrs[v_idx], dists[v_idx])
        else:
            camera_name = input_video_path
            im_names, orig_imgs, im_h, im_w = reader.get_image_frames(camera_name)
        all_view_images[camera_name] = [img[:, :, ::-1] for img in orig_imgs]
        best_frame_id, all_masks = track_and_save_video(args, grounding_processor, grounding_model, video_predictor, f'{sam_text_prompt}.', text_features, len(positive_text_prompts), orig_imgs, open_clip_model, open_clip_preprocess, save_folder=save_segmentation_path, view_name=camera_name, save_video=args.save_seg_video, save_frame=args.save_seg_frame)
        if best_frame_id is None:
            cam_no_masks.append(camera_name)
            print(f'No valid mask at view {camera_name}')
        else:
            all_best_frame_ids.append(best_frame_id)
        all_view_masks[camera_name] = all_masks
    the_best_frame_id = np.asarray(all_best_frame_ids).mean().astype(np.uint8)
    unique_best_frame_ids = np.unique(all_best_frame_ids)
    
    video_predictor.to('cpu')
    del video_predictor
    grounding_model = grounding_model.to('cpu')
    del grounding_model
    if args.to_filter_views:
        open_clip_model.to('cpu')
        del open_clip_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()