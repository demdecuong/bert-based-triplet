import transformers
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import DistilBertTokenizer, DistilBertModel
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from dataloader import create_data_loader, create_triplet_data_loader , get_data_df
from config import get_config
from loss_function import CosineLoss, QuadrupletLoss, TripletLoss
from model import get_model
from evaluate_single import evaluate_model

def train_epoch_aux(model, data_loader, loss_fn, loss_cosine, optimizer , device, config, textwriter):
    model.train()
    losses = []
    loss_func = torch.nn.MarginRankingLoss(0.0)
    for step, batch in enumerate(data_loader):
        # Anchor
        anchor_ids = batch["anchor_ids"].to(device)
        anchor_attention_mask = batch["anchor_attention_mask"].to(device)

        B, max_len = anchor_ids.size()

        anchor_outputs = model(
            input_ids=anchor_ids,
            attention_mask=anchor_attention_mask
        )
        # Positive
        positive_ids = batch["positive_ids"].to(device)
        positive_attention_mask = batch["positive_attention_mask"].to(device)

        positive_outputs = model(
            input_ids=positive_ids,
            attention_mask=positive_attention_mask
        )
        # Negative
        negative_ids = batch["negative_ids"].to(device)
        negative_attention_mask = batch["negative_attention_mask"].to(device)

        negative_outputs = model(
            input_ids=negative_ids,
            attention_mask=negative_attention_mask
        )

        # Triplet Loss
        loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
        
        # Auxilarity Loss 
        ones = torch.ones(B,device = anchor_outputs.device)
        loss_aux1 = loss_func(positive_outputs,negative_outputs,torch.ones((B,config.embed_dim),device = anchor_outputs.device))
        loss_aux2 = loss_cosine(positive_outputs,negative_outputs,ones*-1)
        
        loss = loss + loss_aux1 + loss_aux2        
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % config.print_every == 0:
            print(f"[Train] Loss at step {step} = {loss}, aux1 = {loss_aux1}, aux2 = {loss_aux2}")
            textwriter.write(f"Loss at step {step} = {loss} \n")
    
    return np.mean(losses)

def train_epoch(model, data_loader, loss_fn, optimizer , device, config, textwriter):
    model.train()
    losses = []

    for step, batch in enumerate(data_loader):
        # Anchor
        anchor_ids = batch["anchor_ids"].to(device)
        anchor_attention_mask = batch["anchor_attention_mask"].to(device)

        B, max_len = anchor_ids.size()

        anchor_outputs = model(
            input_ids=anchor_ids,
            attention_mask=anchor_attention_mask
        )
        # Positive
        positive_ids = batch["positive_ids"].to(device)
        positive_attention_mask = batch["positive_attention_mask"].to(device)

        positive_outputs = model(
            input_ids=positive_ids,
            attention_mask=positive_attention_mask
        )
        # Negative
        negative_ids = batch["negative_ids"].to(device)
        negative_attention_mask = batch["negative_attention_mask"].to(device)

        negative_outputs = model(
            input_ids=negative_ids,
            attention_mask=negative_attention_mask
        )
        
        loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if step % config.print_every == 0:
            print(f"[Train] Loss at step {step} = {loss}")
            textwriter.write(f"Loss at step {step} = {loss} \n")
    
    return np.mean(losses)

def eval_model(model , data_loader, loss_fn, device, n_examples, config):
    model.eval()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    correct_predictions = 0
    distances = []
    with torch.no_grad():
        for batch in data_loader:
            question1_ids = batch["question1_ids"].to(device)
            question1_attention_mask = batch["question1_attention_mask"].to(
                device)
            question2_ids = batch["question2_ids"].to(device)
            question2_attention_mask = batch["question2_attention_mask"].to(
                device)
            targets = batch["targets"].to(device)

            question1_outputs = model(
                input_ids=question1_ids,
                attention_mask=question1_attention_mask
            )

            question2_outputs = model(
                input_ids=question2_ids,
                attention_mask=question2_attention_mask
            )

            distance = cos(question1_outputs,question2_outputs)
            distance = torch.mean(distance)
            distances.append(distance.item())   
            
            distance = torch.sum((distance < config.val_threshold).long())

    return distance.item() / n_examples, sum(distances) / len(distances)


if __name__ == '__main__':
    config = get_config()

    np.random.seed(config.seed)

    torch.manual_seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data Loader
    df_train , df_test = get_data_df(config.train_dir, config.val_dir,config)
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    train_data_loader = create_triplet_data_loader(
        df_train, tokenizer, config.max_len, config.batch_size, mode='train')
    test_data_loader = create_data_loader(
        df_test, tokenizer, config.max_len, config.batch_size, mode='val')

    # model
    model = get_model(config)
    model = model.to(device)

    if config.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == 'amsgrad':
        optimizer = torch.optim.Amsgrad(model.parameters(), lr=config.lr)
    elif config.optim == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    elif config.optim == 'adamw':
        optimizer = AdamW(model.parameters(), lr=config.lr, correct_bias=True)
    
    total_steps = len(train_data_loader) * config.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=10000,
        num_training_steps=total_steps
    )
    
    if config.loss_fn == 'triplet':
        loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    elif config.loss_fn == 'cosine' :
        loss_fn = CosineLoss()
    elif config.loss_fn == 'custom_triplet':
        loss_fn = TripletLoss()
    
    if config.use_aux:
      loss_cosine = torch.nn.CosineEmbeddingLoss()
    
    history = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
        'val_loss' : [],
    }

    best_loss = 99999999
    best_top1 = 0
    best_top5 = 0
    best_total = 0
    best_epoch = -1
    config.textfile = open(config.log_dir, "w")

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        print('-' * 10)
        config.textfile.write(f"########## Epoch {epoch} ##########")

        if config.use_aux:
          train_loss = train_epoch_aux(
              model,
              train_data_loader,
              loss_fn,
              loss_cosine,
              optimizer,
              device,
              config,
              config.textfile
            )
        else:
          train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            config,
            config.textfile
          )
        
        print(f'Train loss {train_loss}')

        # val_acc, val_loss = eval_model(
        #     model,
        #     test_data_loader,
        #     loss_fn,
        #     device,
        #     len(df_test),
        #     config
        # )

        top1,top3,top5 = evaluate_model(model,tokenizer,config)

        print(f'Top-1 = {top1} , Top-3 = {top3}, Top-5 = {top5}')
        print()

        history['train_loss'].append(train_loss)
        history['val_acc'].append(top1)

        if top1 + top3 + top5 > best_total:
            print('[SAVE] Saving model ... ')
            torch.save(model.state_dict(), config.model_path)
            best_top1 = top1
            best_top5 = top5
            best_total = top1 + top3 + top5
            best_epoch = epoch
        elif top1 + top3 + top5 == best_top1 and top1 > best_top1:
            print('[SAVE] Saving model ... ')
            torch.save(model.state_dict(), config.model_path)
            best_top1 = top1
            best_total = top1 + top3 + top5
            best_epoch = epoch
        print(f'Best epoch {best_epoch} , Top-1 = {best_top1}')