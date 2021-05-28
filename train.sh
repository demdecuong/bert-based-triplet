python train.py --epochs 15 --batch_size 64 --clip 1.0 --lr 1e-2 --embed_dim 768 \
 --freeze False --space_joiner True --dropout 0.2 \
 --loss_fn triplet --max_len 30 \
 --PRE_TRAINED_MODEL_NAME 'allenai/biomed_roberta_base' 