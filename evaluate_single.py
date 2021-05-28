import torch
import torch.nn as nn
import pandas as pd

from model import get_model
from config import get_config
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

def build_database_embed(model, tokenizer, db, config, device):
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
def compute_topk(model,tokenizer,test_src,test_idx,embed,config,top_k, device):
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
   
    if label_idx in score_list:
        acc += 1
        
  return acc / len(test_idx)

def evaluate_model(model,tokenizer,config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    db = pd.read_csv('./data/QA_database.csv')['question'].tolist()
    test_db = pd.read_csv('./data/test_stroke.csv')
    test_src = test_db['question_x'].tolist()
    test_idx = test_db['idx'].tolist()
    embed = build_database_embed(model,tokenizer,db,config, device)
    test_acc_top5 = compute_topk(model,tokenizer,test_src,test_idx,embed.to(device),config,5,device)
    test_acc_top3 = compute_topk(model,tokenizer,test_src,test_idx,embed.to(device),config,3,device)
    test_acc_top1 = compute_topk(model,tokenizer,test_src,test_idx,embed.to(device),config,1,device)

    return test_acc_top1, test_acc_top3, test_acc_top5
    
if __name__ == '__main__':
    config = get_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    
    db = pd.read_csv('./data/QA_database.csv')['question'].tolist()
    test_db = pd.read_csv('./data/test_stroke.csv')
    test_src = test_db['question_x'].tolist()
    test_idx = test_db['idx'].tolist()

    embed = build_database_embed(model,tokenizer,db,config,device)
    test_acc_top5 = compute_topk(model,tokenizer,test_src,test_idx,embed.to(device),config,5,device)
    test_acc_top3 = compute_topk(model,tokenizer,test_src,test_idx,embed.to(device),config,3,device)
    test_acc_top1 = compute_topk(model,tokenizer,test_src,test_idx,embed.to(device),config,1,device)
    print("Top-5",test_acc_top5)
    print("Top-3",test_acc_top3)
    print("Top-1",test_acc_top1)