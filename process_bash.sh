ROOT_DIR="./dataset/"
OUT_DIR="./dataset/kindle" # "../data/processed"
SESSION="kindle"
IDX_VIDEO="0"
IDX_START="0"
IDX_END="107"
# ANCHOR_CAMERA="brics-odroid-002_cam0"
# ANCHOR_CAMERA="brics-sbc-001_cam0"

# echo "########################## EXTRACT 2D KEYPOINTS ################################"
# python scripts/keypoints_2d_yolo_vitpose.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END \
#         --use_hamer True --use_optim_params # --sample_fps 30 --frame_count 300

# echo "######################### TRIANGULATE 3D KEYPOINTS #########################"
# python scripts/keypoints_3d_fast.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
#         --undistort --all_frames --easymocap --use_optim_params --to_smooth

echo "################################ MANO FIT ##################################"
python scripts/mano_em.py -r $ROOT_DIR -s $SESSION -o $OUT_DIR \
        --model manor --body handr --undistort  --ith $IDX_VIDEO --start $IDX_START --end $IDX_END\
        --use_filtered --use_optim_params --stride 1 --to_smooth \
        --vis_smpl --vis_2d_repro --vis_3d_repro