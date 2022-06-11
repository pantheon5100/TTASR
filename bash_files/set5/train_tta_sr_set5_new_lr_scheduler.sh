
python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "TTA_set5-cos_lr-iter_1000-lr2e6" \
        --batch_size 16 \
        --num_iters 1000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 500 \
        --model_save_iter 1000 \
        --eval_iters 2 \
        --lr_G_UP 2e-7 \
        --lr_G_DN 0.001 \
        --update_l_rate_freq 100

# python tta_main.py \
#         --input_dir "test/Set5/LR_bicubic/X2" \
#         --gt_dir "test/Set5/HR" \
#         --output_dir "TTA_set5-cos_lr-iter_2000-lr2e8" \
#         --num_iters 2000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 1000 \
#         --model_save_iter 2000 \
#         --eval_iters 2 \
#         --lr_G_UP 2e-8 \
#         --lr_G_DN 0.001 \
#         --update_l_rate_freq 200


# python tta_main.py \
#         --input_dir "test/Set5/LR_bicubic/X2" \
#         --gt_dir "test/Set5/HR" \
#         --output_dir "TTA_set5-cos_lr-iter_4000-lr2e6" \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --model_save_iter 4000 \
#         --eval_iters 2 \
#         --lr_G_UP 2e-6 \
#         --lr_G_DN 0.001



# python tta_main.py \
#         --input_dir "test/Set5/LR_bicubic/X2" \
#         --gt_dir "test/Set5/HR" \
#         --output_dir "TTA_set5-cos_lr-iter_4000-lr2e7" \
#         --num_iters 4000 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 3000 \
#         --model_save_iter 4000 \
#         --eval_iters 2 \
#         --lr_G_UP 2e-7 \
#         --lr_G_DN 0.001



python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "TTA_set5-cos_lr-iter_4000-lr2e8" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --model_save_iter 4000 \
        --eval_iters 2 \
        --lr_G_UP 2e-8 \
        --lr_G_DN 0.001


python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "TTA_set5-cos_lr-iter_4000-lr2e6" \
        --num_iters 2500 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 1500 \
        --model_save_iter 2500 \
        --eval_iters 2 \
        --lr_G_UP 2e-6 \
        --lr_G_DN 0.001 \
        --update_l_rate_freq 325
