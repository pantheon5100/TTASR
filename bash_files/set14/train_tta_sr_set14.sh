

python tta_main.py \
        --input_dir "../dataset/Set14/LR_bicubic/x2" \
        --gt_dir "../dataset/Set14/HR" \
        --output_dir "TTA_set14" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --model_save_iter 4000 \
        --eval_iters 2 \
        --lr_G_UP 2e-6 \
        --lr_G_DN 0.001


