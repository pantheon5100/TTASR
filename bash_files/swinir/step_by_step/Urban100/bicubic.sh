

# # Set5
# python tta_main.py \
#         --input_dir "test/Set5/LR_bicubic/X2" \
#         --gt_dir "test/Set5/HR" \
#         --output_dir "TTA_sbs-bicubic" \
#         --train_mode "bicubic" \
#         --num_iters 1000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 1 \
#         --model_save_iter 100 \
#         --eval_iters 2 \
#         --lr_G_UP 4e-7 \
#         --lr_G_DN 0.01 \
#         --update_l_rate_freq_gdn 10 \
#         --update_l_rate_freq_gup 100


# BSD100
python tta_main.py \
        --input_dir "../dataset/Urban100/LR_bicubic/x2" \
        --gt_dir "../dataset/Urban100/HR" \
        --output_dir "TTA_sbs-bicubic-Urban100" \
        --train_mode "bicubic" \
        --num_iters 1000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 1 \
        --model_save_iter 100 \
        --eval_iters 2 \
        --lr_G_UP 4e-7 \
        --lr_G_DN 0.01 \
        --update_l_rate_freq_gdn 10 \
        --update_l_rate_freq_gup 100

