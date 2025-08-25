python train_diffusion.py \
    --train_paths "./data/train.npz" \
    --epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --best_model_path "./pretrain_weight/diffusion_unet.pth" \
    --dim 2048 \
    --channels_num 1 