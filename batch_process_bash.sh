# source ~/.bashrc
# conda activate /users/rfu7/data/anaconda/object_env
# cd object_estimation

# ITH=$1
# SESSION=$2
# SCANPATH=$3

ITH=1
SESSION="rubiks_cube"
# SCANPATH=$3



echo "########################## TRACKING OBJECT POSE ################################"
python scripts/1_object_tracking.py \
    --ith $ITH \
    --session $SESSION \
    --save_seg_frame \
    --step1_only \
    --sample_fps 30 \
    --frame_count 300
    # --scan_path $SCANPATH \