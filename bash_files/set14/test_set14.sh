python tta_main.py \
        --input_dir "../dataset/Set14/LR_bicubic/x2" \
        --gt_dir "../dataset/Set14/HR" \
        --output_dir "test_swinir_set14" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --test_only

