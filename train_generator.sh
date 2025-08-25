python generator/train_ECGLDM.py \
    --train_data_path "./data/train.npz" \
    --epoches 200 \
    --batch_size 32 \
    --lr 1e-4 \
    --best_model_path "./pretrain_weight/best_model.pth" \
    --dim 2048 \
    --channels_num 1 