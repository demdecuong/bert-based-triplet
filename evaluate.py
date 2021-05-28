import torch
import torch.nn as nn
import pandas as pd

from model import get_model
from config import get_config
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

def build_database_embed(model, tokenizer, db, config):
    # Build embedding space for DB
    embed = torch.zeros((len(db),config.embed_dim))
    for i,sent in enumerate(db):
        encoding = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length= config.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            # padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        # print(encoding['attention_mask'])
        output = model(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device)
        )
        embed[i] = output
    return embed

from tqdm import tqdm
def compute_topk(model,tokenizer,test_src,test_idx,embed,config,top_k=1):
  acc = 0
  for source, label_idx in tqdm(zip(test_src,test_idx)):
    score_list = []
    encoding = tokenizer.encode_plus(
      source,
      add_special_tokens=True,
      max_length= config.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      # padding='max_length',
      return_attention_mask=True,
      return_tensors='pt',
      truncation=True,
    )
    # print(encoding['attention_mask'])
    output = model(
        input_ids=encoding['input_ids'].to(device),
        attention_mask=encoding['attention_mask'].to(device)
      )
    
    output = output.repeat(200,1)
    cosine = nn.CosineSimilarity(dim=1, eps=1e-6)    # distance = (output - embed) ** 2
    distance = cosine(output,embed)
    # distance = torch.mean(distance,1)
    score_list = torch.topk(distance,top_k).indices.tolist()
    # print(torch.topk(distance,top_k).values)
    candidates = sorted(range(len(score_list)),key=score_list.__getitem__,reverse=True)[:top_k]
    # print(sorted(score_list,reverse=True)[:top_k])
    # print(candidates,label_idx)
    if label_idx in candidates:
        acc += 1
        
  return acc / len(test_idx)

if __name__ == '__main__':
    config = get_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    model_anchor = get_model(config).to(device)
    model_pos_neg = get_model(config).to(device)
    model_anchor.load_state_dict(torch.load(config.model_path))
    model_pos_neg.load_state_dict(torch.load(config.model_path + '_pos_neg'))
    model_anchor.eval()
    model_pos_neg.eval()

    db = pd.read_csv('./data/QA_database.csv')['question'].tolist()
    test_db = pd.read_csv('./data/test_stroke.csv')
    test_src = test_db['question_x'].tolist()
    test_idx = test_db['idx'].tolist()

    embed = build_database_embed(model_pos_neg,tokenizer,db,config)
    test_acc_top5 = compute_topk(model_anchor,tokenizer,test_src,test_idx,embed.to(device),config,top_k=5)
    test_acc_top3 = compute_topk(model_anchor,tokenizer,test_src,test_idx,embed.to(device),config,top_k=3)
    test_acc_top1 = compute_topk(model_anchor,tokenizer,test_src,test_idx,embed.to(device),config,top_k=1)
    print("Top-5",test_acc_top5)
    print("Top-3",test_acc_top5)
    print("Top-1",test_acc_top5)