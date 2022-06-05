# python tta_main_test_swinir.py \
#     --task classical_sr \
#     --scale 2 \
#     --training_patch_size 48 \
#     --model_path tta_pretrained/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \
#     --folder_lq test/Set5/LR_bicubic/X2 \
#     --folder_gt test/Set5/HR


python tta_main_test_swinir.py \
    --task classical_sr \
    --scale 2 \
    --training_patch_size 48 \
    --model_path tta_pretrained/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth \
    --folder_lq "test/Set14/LR_bicubic/X2" \
    --folder_gt "test/Set14/HR" \
    --test_only
