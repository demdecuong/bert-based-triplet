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

def train_epoch(model_anchor, model_pos_neg, data_loader, loss_fn, optimizer_a, optimizer_pn, device, config, textwriter):
    model_anchor.train()
    model_pos_neg.train()
    losses = []

    for step, batch in enumerate(data_loader):
        # Anchor
        anchor_ids = batch["anchor_ids"].to(device)
        anchor_attention_mask = batch["anchor_attention_mask"].to(device)

        anchor_outputs = model_anchor(
            input_ids=anchor_ids,
            attention_mask=anchor_attention_mask
        )
        # Positive
        positive_ids = batch["positive_ids"].to(device)
        positive_attention_mask = batch["positive_attention_mask"].to(device)

        positive_outputs = model_pos_neg(
            input_ids=positive_ids,
            attention_mask=positive_attention_mask
        )
        # Negative
        negative_ids = batch["negative_ids"].to(device)
        negative_attention_mask = batch["negative_attention_mask"].to(device)

        negative_outputs = model_pos_neg(
            input_ids=negative_ids,
            attention_mask=negative_attention_mask
        )

        loss = loss_fn(anchor_outputs, positive_outputs, negative_outputs)
        losses.append(loss.item())
        loss.backward()

        nn.utils.clip_grad_norm_(model_anchor.parameters(), max_norm=config.clip)
        nn.utils.clip_grad_norm_(model_pos_neg.parameters(), max_norm=config.clip)

        optimizer_a.step()
        optimizer_pn.step()
        # scheduler.step()
        optimizer_a.zero_grad()
        optimizer_pn.zero_grad()

        if step % config.print_every == 0:
            print(f"[Train] Loss at step {step} = {loss}")
            textwriter.write(f"Loss at step {step} = {loss} \n")
    return np.mean(losses)


def eval_model(model_anchor, model_pos_neg, data_loader, loss_fn, device, n_examples, config):
    model_anchor.eval()
    model_pos_neg.eval()
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

            question1_outputs = model_anchor(
                input_ids=question1_ids,
                attention_mask=question1_attention_mask
            )

            question2_outputs = model_pos_neg(
                input_ids=question2_ids,
                attention_mask=question2_attention_mask
            )

            distance = (question1_outputs - question2_outputs).pow(2).sum(1)
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
    df_train , df_test = get_data_df(config.train_dir, config.val_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    train_data_loader = create_triplet_data_loader(
        df_train, tokenizer, config.max_len, config.batch_size, mode='train')
    test_data_loader = create_data_loader(
        df_test, tokenizer, config.max_len, config.batch_size, mode='val')

    # model
    model_anchor = get_model(config)
    model_pos_neg = get_model(config)
    model_anchor = model_anchor.to(device)
    model_pos_neg = model_pos_neg.to(device)

    if config.optim == 'adam':
        optimizer_a = torch.optim.Adam(model_anchor.parameters(), lr=config.lr)
        optimizer_pn = torch.optim.Adam(model_pos_neg.parameters(), lr=config.lr)
    elif config.optim == 'amsgrad':
        optimizer_a = torch.optim.Amsgrad(model_anchor.parameters(), lr=config.lr)
        optimizer_pn = torch.optim.Amsgrad(model_pos_neg.parameters(), lr=config.lr)
    elif config.optim == 'adagrad':
        optimizer_a = torch.optim.Adagrad(model_anchor.parameters(), lr=config.lr)
        optimizer_pn = torch.optim.Adagrad(model_pos_neg.parameters(), lr=config.lr)
    elif config.optim == 'adamw':
        optimizer_a = AdamW(model_anchor.parameters(), lr=config.lr, correct_bias=True)
        optimizer_pn = AdamW(model_pos_neg.parameters(), lr=config.lr, correct_bias=True)

    total_steps = len(train_data_loader) * config.epochs

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps
    # )
    
    if config.loss_fn == 'triplet':
        loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
    elif config.loss_fn == 'cosine' :
        loss_fn = CosineLoss()
    elif config.loss_fn == 'custom_triplet':
        loss_fn = TripletLoss()

    history = {
        'train_acc' : [],
        'train_loss' : [],
        'val_acc' : [],
        'val_loss' : [],
    }

    best_loss = 99999999
    config.textfile = open(config.log_dir, "w")

    for epoch in range(config.epochs):
        print(f'Epoch {epoch + 1}/{config.epochs}')
        print('-' * 10)
        config.textfile.write(f"########## Epoch {epoch} ##########")

        train_loss = train_epoch(
            model_anchor,
            model_pos_neg,
            train_data_loader,
            loss_fn,
            optimizer_a,
            optimizer_pn,
            device,
            config,
            config.textfile
        )
        
        print(f'Train loss {train_loss}')

        val_acc, val_loss = eval_model(
            model_anchor,
            model_pos_neg,
            test_data_loader,
            loss_fn,
            device,
            len(df_test),
            config
        )

        print(f'Val mean distance {val_loss} accuracy {val_acc}')
        print()

        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_loss < best_loss:
            print('[SAVE] Saving model ... ')
            torch.save(model_anchor.state_dict(), config.model_path)
            torch.save(model_pos_neg.state_dict(), config.model_path + '_pos_neg')
            best_loss = val_loss