
python tta_main.py \
        --input_dir "../dataset/BSD/LR_bicubic/x2" \
        --gt_dir "../dataset/BSD/HR" \
        --output_dir "TTA_BSD" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP 2e-6 \
        --lr_G_DN 0.001

