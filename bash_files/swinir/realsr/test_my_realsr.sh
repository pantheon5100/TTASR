python tta_main.py \
        --input_dir "test/my_RealSR/Test/2/LR" \
        --gt_dir "test/my_RealSR/Test/2/HR" \
        --output_dir "test_swinir_real_SR" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 1 \
        --lr_G_UP 0.00002\
        --lr_G_DN 0.001 \
        --test_only

