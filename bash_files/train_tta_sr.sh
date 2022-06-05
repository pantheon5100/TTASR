# python tta_main.py --input_dir "test/Set5/LR_bicubic/X2" --input_crop_size 48



# python tta_main.py \
#         --input_dir "test/Set5/LR_bicubic/X2" \
#         --gt_dir "test/Set5/HR" \
#         --output_dir "longer_train" \
#         --num_iters 6000 \
#         --switch_iters 1000 \
#         --input_crop_size 48 \
#         --scale_factor 4 \
#         --lr_G_UP 0.0002 \
#         --lr_G_DN 0.001


# python tta_main.py \
#         --input_dir "test/Set5/LR_bicubic/X2" \
#         --gt_dir "test/Set5/HR" \
#         --output_dir "frequence_switch" \
#         --num_iters 1100 \
#         --input_crop_size 48 \
#         --scale_factor 2 \
#         --switch_iters 1000 \
#         --eval_iters 1 \
#         --lr_G_UP 0.00002\
#         --lr_G_DN 0.001


python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "frequence_switch" \
        --num_iters 8000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 6000 \
        --eval_iters 1 \
        --lr_G_UP 0.0002\
        --lr_G_DN 0.01

