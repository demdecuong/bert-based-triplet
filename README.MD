## Sentence Matching with BERT-based and triplet loss

![Overview](./asset/asset.png)

### Installation
    pip install -r requirement.txt

### Usage

**Train**
```
python train.py --epochs 15 --batch_size 64 --clip 1.0 --lr 1e-2 --embed_dim 100
```

**TestDB**
```
python evaluate.py --model_path ./ckpt/best_model_state_v1_triplet.bin 
```


TODO  
Train MLM 