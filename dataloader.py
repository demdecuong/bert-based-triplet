import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class GPReviewDataset(Dataset):

    def __init__(self, question1, question2, targets, tokenizer, max_len):
        self.question1 = question1
        self.question2 = question2
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.question1)

    def __getitem__(self, item):

        question1 = str(self.question1[item])
        question2 = str(self.question2[item])
        target = self.targets[item]

        question1_encoding = self.tokenizer.encode_plus(
            question1,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        question2_encoding = self.tokenizer.encode_plus(
            question2,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'question1': question1,
            'question2': question2,
            'question1_ids': question1_encoding['input_ids'].flatten(),
            'question1_attention_mask': question1_encoding['attention_mask'].flatten(),
            'question2_ids': question2_encoding['input_ids'].flatten(),
            'question2_attention_mask': question2_encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class TripletDataset(Dataset):

    def __init__(self, anchor, positive, negative, tokenizer, max_len):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.anchor)

    def __getitem__(self, item):

        anchor = str(self.anchor[item])
        positive = str(self.positive[item])
        negative = str(self.negative[item])

        anchor_encoding = self.tokenizer.encode_plus(
            anchor,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        positive_encoding = self.tokenizer.encode_plus(
            positive,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        negative_encoding = self.tokenizer.encode_plus(
            negative,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        return {
            'anchor': anchor,
            'positive': positive,
            'negative': negative,
            'anchor_ids': anchor_encoding['input_ids'].flatten(),
            'anchor_attention_mask': anchor_encoding['attention_mask'].flatten(),
            'positive_ids': positive_encoding['input_ids'].flatten(),
            'positive_attention_mask': positive_encoding['attention_mask'].flatten(),
            'negative_ids': negative_encoding['input_ids'].flatten(),
            'negative_attention_mask': negative_encoding['attention_mask'].flatten(),
        }


def create_data_loader(df, tokenizer, max_len, batch_size, mode='train'):
    ds = GPReviewDataset(
        question1=df.question_x.to_numpy(),
        question2=df.question_y.to_numpy(),
        targets=df.is_duplicate.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    if mode == 'train':
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )


def create_triplet_data_loader(df, tokenizer, max_len, batch_size, mode='train'):
    ds = TripletDataset(
        anchor=df.anchor.to_numpy(),
        positive=df.positive.to_numpy(),
        negative=df.negative.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    if mode == 'train':
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
    else:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        )

def get_data_df(train_dir,test_dir):
    df_test = pd.read_csv(test_dir)
    df_test['is_duplicate'] = [1] * len(df_test)
    # df_test2 = pd.read_csv('../test_final.csv')
    # df_test = df_test[['question1','question2','is_duplicate']]
    # df_test = pd.concat([df_test,df_test2])


    # Train data
    # df_train = pd.read_csv('../test_final.csv') #
    # df_train2 = pd.read_csv('../3254_train.csv') #
    # df_train3 = pd.read_csv('../600_train.csv') #
    # df_train3 = df_train3.drop('Unnamed: 0',axis=1)
    # df_train4 = pd.read_csv('../9303_train.csv') #
    # df_train4.sample(frac=1)
    # df_train4 = df_train4[:3000]
    # df_train = pd.concat([df_train,df_train2,df_train3,df_train4])

    #Train data triplet

    df_train = pd.read_csv(train_dir)
    df_train.dropna()
    df_train = df_train.drop('Unnamed: 0',axis=1)

    df_train.shape, df_test.shape # question1, question2, is_duplicate
    return df_train, df_test