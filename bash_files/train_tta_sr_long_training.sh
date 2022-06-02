# python tta_main.py --input_dir "test/Set5/LR_bicubic/X2" --input_crop_size 48



python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --input_crop_size 48 \
        --lr_G_UP 0.0002\
        --lr_G_DN 0.001
