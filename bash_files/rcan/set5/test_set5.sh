python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "test_swinir_set5" \
        --source_model "rcan" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 1 \
        --lr_G_UP 0.00002\
        --lr_G_DN 0.001 \
        --test_only

