
python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "TTA_set5-image_agnostic_gdn" \
        --train_mode "image_agnostic_gdn" \
        --pretrained_GDN "log/TTA_set5-image_agnostic_gdn/time_20220612081100lr_GUP_2e-07-lr_GDN_0.001input_size_48-scale_factor_2/ckpt/pretrained_GDN.ckpt"\
        --num_iters 500 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 1 \
        --model_save_iter 500 \
        --eval_iters 2 \
        --lr_G_UP 4e-7 \
        --lr_G_DN 0.01 \
        --update_l_rate_freq_gdn 10 \
        --update_l_rate_freq_gup 100



