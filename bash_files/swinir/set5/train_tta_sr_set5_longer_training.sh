

python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "TTA_set5_longer_training-in48-bs2" \
        --batch_size 2 \
        --num_iters 8000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 6000 \
        --model_save_iter 4000 \
        --eval_iters 2 \
        --lr_G_UP 2e-6 \
        --lr_G_DN 0.001


python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "TTA_set5_longer_training-in96-bs8" \
        --batch_size 8 \
        --num_iters 8000 \
        --input_crop_size 96 \
        --scale_factor 2 \
        --switch_iters 6000 \
        --model_save_iter 4000 \
        --eval_iters 2 \
        --lr_G_UP 2e-6 \
        --lr_G_DN 0.001

