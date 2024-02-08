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
from language_processing import collate_fn, train_dataset, valid_dataset, vocab_transform
from model import Transformer

# hyperparameters
batch_size = 192 # how many independent sequences will we process in parallel?
block_size = 100 # what is the maximum context length for predictions?
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
# we expect head_size to be 16
n_layer = 4
dropout = 0.0
# ------------

# language parameters
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
src_vocab_size = len(vocab_transform[SRC_LANGUAGE])
tgt_vocab_size = len(vocab_transform[TGT_LANGUAGE])
# ------------

def train_epoch(model, optimizer, train_dataloader):
    print('Training')
    model.train()
    losses = 0

    # note that below gets batches 
    for src, tgt in train_dataloader:            
        # print(" ".join(vocab_transform[SRC_LANGUAGE].lookup_tokens(list(src[0].cpu().numpy()))).replace("<bos>", "").replace("<eos>", ""))
        # print(" ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt[0].cpu().numpy()))).replace("<bos>", "").replace("<eos>", ""))
        src = src.to(device)
        tgt = tgt.to(device)
                
        tgt_input = tgt[:, :-1]
        
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(
            tgt_input,
            src
        )
        optimizer.zero_grad() # just need to do this before calling backward (gradients not updated in forward or loss I think)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.view(-1, tgt_vocab_size), tgt_out.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(list(train_dataloader)) # if train_dataloader returns batches, then this returns loss per batch

val_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)

def evaluate(model, val_dataloader):
    print('Validating')
    model.eval()
    losses = 0
    for src, tgt in val_dataloader:
        # print(" ".join(vocab_transform[SRC_LANGUAGE].lookup_tokens(list(src[0].cpu().numpy()))).replace("<bos>", "").replace("<eos>", ""))
        # print(" ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt[0].cpu().numpy()))).replace("<bos>", "").replace("<eos>", ""))
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        logits = model(
            tgt_input,
            src
        )
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.view(-1, tgt_vocab_size), tgt_out.contiguous().view(-1))
        losses += loss.item()
    return losses / len(list(val_dataloader))

transformer = Transformer(n_embd, n_head, n_layer, dropout, src_vocab_size, tgt_vocab_size, SRC_LANGUAGE, TGT_LANGUAGE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
train_loss_list, valid_loss_list = [], []
for epoch in range(1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    valid_loss = evaluate(transformer)
    end_time = timer()
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {valid_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s \n"))