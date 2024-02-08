from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer as timer
from torch.nn import Transformer
from torch import Tensor
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import os
import pandas as pd
import matplotlib.pyplot as plt

SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'

token_transform = {}
vocab_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='fr_core_news_sm')

df = pd.read_csv(
    'data/eng-french.csv', 
    usecols=['English words/sentences', 'French words/sentences']
)

train_df, test_df = train_test_split(df, test_size=0.1)

# Custom Dataset class.
class TranslationDataset(Dataset):
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return(
            self.df['English words/sentences'].iloc[idx], # iloc gets row
            self.df['French words/sentences'].iloc[idx]
        )

train_dataset = TranslationDataset(train_df)
valid_dataset = TranslationDataset(test_df)

# Helper function to yield list of tokens.
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]]) 
    # say we feed in train_dataset and "eng"
    # then for each entry in train_dataset, this yields the tokenized english part of the data
    # token_transform[language] gets the appropriate tokenizer
    # data_sample will be data_sample[0] or data_sampe[1] and will be the input we wish to have tokenized

# Define special symbols and indices.
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Unknown words, padding, beginning of sequence, end of sequence

# Make sure the tokens are in order of their indices to properly insert them in vocab.
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
def create_vocab_transform():
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Create torchtext's Vocab object.
        vocab_transform[ln] = build_vocab_from_iterator( # this maps each token to a unique iterator
            yield_tokens(train_dataset, ln),
            min_freq=1,
            specials=special_symbols,
            special_first=True,
        )

    # Set ``UNK_IDX`` as the default index. This index is returned when the token is not found.
    # If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

vocab_transform = create_vocab_transform()

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))
    
# `src` and `tgt` language text transforms to convert raw strings into tensors indices
# it does this by transforming it into tokens, getting integers for tokens (from for loop initializing vocab_transform),
# and adding BOS and EOS from above
def create_text_transform():
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                                vocab_transform[ln], # Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor

text_transform = create_text_transform()

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch