import torch
import torch.nn as nn
import pandas as pd

from model import get_model
from config import get_config
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class Inferencer:
  def __init__(self,config):
    self.db = pd.read_csv('./data/QA_database.csv')['question'].tolist()
    self.answer = pd.read_csv('./data/QA_database.csv')['answer'].tolist()

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    self.tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    self.model = get_model(config).to(self.device)
    self.model.load_state_dict(torch.load(config.model_path))
    self.model.eval()

    self.embed = self.build_database_embed()
    self.cosine = nn.CosineSimilarity(dim=1, eps=1e-6)    # distance = (output - embed) ** 2

  def build_database_embed(self):
      # Build embedding space for DB
      embed = torch.zeros((len(self.db),config.embed_dim)).to(self.device)
      for i,sent in enumerate(self.db):
          encoding = self.tokenizer.encode_plus(
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
          output =self. model(
              input_ids=encoding['input_ids'].to(self.device),
              attention_mask=encoding['attention_mask'].to(self.device)
          )
          embed[i] = output
      return embed

  def predict(self,question,top_k=3):
      score_list = []
      encoding = self.tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length= config.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        # padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
      )
      
      output = self.model(
          input_ids=encoding['input_ids'].to(self.device),
          attention_mask=encoding['attention_mask'].to(self.device)
        )
      
      output = output.repeat(len(self.db),1)
      distance = self.cosine(output,self.embed)
      # distance = torch.mean(distance,1)
      score_list = torch.topk(distance,top_k).indices.tolist()
      confidence_list = torch.topk(distance,top_k).values.tolist()
      result = []
      for idx,conf in zip(score_list,confidence_list):
        result.append( (self.db[idx],conf,self.answer[idx]))
      return result

if __name__ == '__main__':
    config = get_config()

    inferencer = Inferencer(config)

    result = inferencer.predict('What is red stroke')
    print(result)