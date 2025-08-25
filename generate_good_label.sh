python generator/generate.py \
    --channels_num 1 \
    --save_path "./generated/generated_good_data.npz" \
    --checkpoint_path "./pretrain_weight/diffusion_unet.pth" \
    --generate_num 100 \
    --iterations 2 \
    --label 0 \
    --dim 2048 