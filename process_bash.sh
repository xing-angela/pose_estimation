# ROOT_DIR="./dataset"
# OUT_DIR="./dataset/soda"
# SESSION="soda"

# python3 scripts/0_generate_ingp_template.py \
#     --root_dir $ROOT_DIR \
#     --out_dir $OUT_DIR \
#     --seq_path $SESSION \
#     --use_optim_params

# ROOT_DIR="/users/axing2/data/public/manip"
# OUT_DIR="./dataset/book_gt1"
# SESSION="2025-01-08"
# ANCHOR_CAMERA="brics-odroid-001_cam0"
# TEXT_PROMPT="hand"

# python3 scripts/gt/1_object_segment.py \
#     --ith 14 \
#     --root_dir $ROOT_DIR \
#     --out_dir $OUT_DIR \
#     --session $SESSION \
#     --use_optim_params \
#     --save_seg_frame \
#     --anchor_camera $ANCHOR_CAMERA \
#     --text_prompt $TEXT_PROMPT

# ROOT_DIR="/dataset"
# OUT_DIR="./dataset/book_gt1"
# SESSION="2025-01-08"
# ANCHOR_CAMERA="brics-odroid-001_cam0"

# python3 scripts/gt/2_ngp_mesh.py \
#     --ith 14 \
#     --root_dir $ROOT_DIR \
#     --out_dir $OUT_DIR \
#     --session $SESSION \
#     --use_optim_params \
#     --anchor_camera $ANCHOR_CAMERA

ROOT_DIR="/users/axing2/data/public/manip"
OUT_DIR="./dataset/book_gt1"
SESSION="2025-01-08"
ANCHOR_CAMERA="brics-odroid-001_cam0"
TEXT_PROMPT="book"
SCAN_PATH="book_simplified.obj"

python3 scripts/gt/3_track_object.py \
    --ith 14 \
    --root_dir $ROOT_DIR \
    --out_dir $OUT_DIR \
    --session $SESSION \
    --use_optim_params \
    --anchor_camera $ANCHOR_CAMERA \
    --text_prompt $TEXT_PROMPT \
    --scan_path $SCAN_PATH