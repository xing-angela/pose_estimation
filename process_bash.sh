# ROOT_DIR="/users/axing2/data/brics/non-pii/brics-mini/"
# OUT_DIR="/users/axing2/data/users/axing2/hand_pose_estimation/dataset/2024-12-10" # "../data/processed"
# SESSION="2024-12-10"
ROOT_DIR="./dataset/"
OUT_DIR="./dataset/rubiks_cube" # "../data/processed"
SESSION="rubiks_cube"
IDX_VIDEO="0"
IDX_START="0"
IDX_END="-1"
# ANCHOR_CAMERA="brics-odroid-002_cam0"
ANCHOR_CAMERA="brics-sbc-001_cam0"

echo "########################## EXTRACT 2D KEYPOINTS ################################"
python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END \
        --anchor_camera $ANCHOR_CAMERA --use_hamer True --use_optim_params --sample_fps 30 --frame_count 300

echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
python scripts/keypoints_3d_fast.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
        --undistort --all_frames --easymocap --use_optim_params --to_smooth  --anchor_camera $ANCHOR_CAMERA --sample_fps 30\
        --frame_count 300

echo "################################ MANO FIT ##################################"
python scripts/mano_em.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR \
        --model manor --body handr --undistort  --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
        --use_filtered --use_optim_params --stride 1 --to_smooth  --anchor_camera $ANCHOR_CAMERA \
        --vis_smpl --vis_2d_repro --vis_3d_repro --sample_fps 30 --frame_count 300 # --save_mesh --save_frame--