
GUP_LEARNINGRATE="2e-4 2e-5 2e-6 2e-7 2e-8 2e-9 2e-10"
GDN_LEARNINGRATE="2e-1 2e-2 2e-3 2e-4 2e-5 2e-6"


for gup_learningrate in $GUP_LEARNINGRATE
do

for gdn_learningrate in $GDN_LEARNINGRATE
do
python tta_main.py \
        --input_dir "test/Set5/LR_bicubic/X2" \
        --gt_dir "test/Set5/HR" \
        --output_dir "learning_rate_search_set5" \
        --num_iters 4000 \
        --input_crop_size 48 \
        --scale_factor 2 \
        --switch_iters 3000 \
        --eval_iters 2 \
        --lr_G_UP $gup_learningrate\
        --lr_G_DN $gdn_learningrate

done
done
