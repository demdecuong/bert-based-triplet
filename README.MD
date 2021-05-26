## Sentence Matching with BERT-based and triplet loss

![Overview](./asset/asset.png)

### Installation
    pip install -r requirement.txt

### Usage

**Train**
```
python train.py --epochs 15 --batch_size 128 --clip 1.0 --lr 1e-3 --embed_dim 300 \
 --freeze False --space_joiner True --dropout 0.2 \
 --loss_fn triplet --max_len 30 \
 --PRE_TRAINED_MODEL_NAME 'allenai/biomed_roberta_base' \
 --model_path './ckpt/best_model_state_v2_triplet' \
 --train_dir './data/100k_data.csv'
```

**TestDB**
```
python evaluate.py --embed_dim 300 \
 --freeze False --space_joiner True --dropout 0.2 \
 --max_len 30 \
 --PRE_TRAINED_MODEL_NAME 'allenai/biomed_roberta_base' \
 --model_path './ckpt/best_model_state_v2_triplet'
```


TODO  
Train MLM 