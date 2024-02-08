import torch
import torch.nn.functional as F
import torch.nn as nn

max_len = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# this gets shapes right but doesn't seem to work, I'm not sure why
class MultiHeadAttention(nn.Module):
    '''
    Multi Head Self Attention module. Initialize with embedding dimension and numbers of heads.
    Masked is a boolean which determines if a mask is applied that prevents looking forward at future tokens.
    A padding mask can also be applied in the forward method which will prevent attention on padding tokens.
    Note that we must have n_embd % n_heads = 0 so that n_heads * head_size = n_embd
    '''
    def __init__(self, n_embd, n_heads, dropout, masked=False):
        super().__init__()
        self.attn = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_size = n_embd // n_heads
        self.masked = masked
        if masked:
          self.register_buffer('tril', torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        if self.masked:
          wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        if padding_mask is not None:
          wei = wei.masked_fill(padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))

        wei = F.softmax(wei, dim=-1)
        out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    """ Encoder block: communication followed by computation """

    def __init__(self, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout, False)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, src, src_padding_mask):
        src = src + self.sa(self.ln1(src), src_padding_mask)
        src = src + self.ffwd(self.ln2(src))
        return src

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.Sequential(*[EncoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm

    def forward(self, x, src_padding_mask):
        B, T = x.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(x) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
          x = block(x, src_padding_mask) # (B,T,C)
        y = self.ln_f(x) # (B,T,C)

        return y

class EncoderDecoder(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.attn = nn.Linear(n_embd, 3*n_embd, bias=False)
        self.key_value = nn.Linear(n_embd, 2*n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(self, tgt, src):
        B, T, C = tgt.shape
        _, J, _ = src.shape
        k, v = self.key_value(src).split(self.n_embd, dim=2)
        q = self.query(tgt)
        k = k.view(B, J, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, J, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        out = self.dropout(self.proj(out))
        return out

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout, True)
        self.ed = EncoderDecoder(n_embd, n_head, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, tgt, src, tgt_padding_mask):
        result = tgt + self.sa(self.ln1(tgt), tgt_padding_mask)
        result = result + self.ed(self.ln2(result), self.ln2(src)) # not sure if this is the right way to apply layer norm
        result = result + self.ffwd(self.ln3(tgt))
        return result

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, e, tgt_padding_mask):
        B, T = x.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(x) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        for block in self.blocks:
          x = block(x, e, tgt_padding_mask)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        return logits

    # def generate(self, idx, max_new_tokens):
    #     # idx is (B, T) array of indices in the current context
    #     for _ in range(max_new_tokens):
    #         # crop idx to the last block_size tokens
    #         idx_cond = idx[:, -block_size:]
    #         # get the predictions
    #         logits, loss = self(idx_cond)
    #         # focus only on the last time step
    #         logits = logits[:, -1, :] # becomes (B, C)
    #         # apply softmax to get probabilities
    #         probs = F.softmax(logits, dim=-1) # (B, C)
    #         # sample from the distribution
    #         idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
    #         # append sampled index to the running sequence
    #         idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    #     return idx

class Transformer(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, dropout, src_vocab_size, tgt_vocab_size, src_language, tgt_language):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, n_embd, n_head, n_layer, dropout)
        self.decoder = Decoder(tgt_vocab_size, n_embd, n_head, n_layer, dropout)
        self.src_language = src_language
        self.tgt_language = tgt_language

    def forward(self, x, src, src_padding_mask, tgt_padding_mask):
        # src_padding_mask = (src == PAD_IDX)
        # tgt_padding_mask = (x == PAD_IDX)
        e_src = self.encoder(src, src_padding_mask)
        out = self.decoder(x, e_src, tgt_padding_mask)
        return out

class Translator:
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

    def __init__(self, model, text_transform, vocab_transform):
        self.model = model
        self.text_transform = text_transform
        self.vocab_transform = vocab_transform
        self.input_language = self.model.src_language
        self.output_language = self.model.tgt_language
    
    # the theory behind below may be totally wrong, this is my guess
    def decode(self, src, max_len):
        # in a translate function we will convert a sentence into src
        # this will then decode it
        B, T = src.shape
        src_padding_mask = torch.zeros_like(src, device=device).type(torch.bool)
        e_src = self.encoder(src, src_padding_mask) # need to add a padding mask to this now
        target = torch.ones(B, 1, device=device).fill_(self.BOS_IDX).type(torch.long)
        for i in range(max_len-1):
            temp = target[:, -1].unsqueeze(1)
            tgt_padding_mask = torch.zeros_like(temp, device=device).type(torch.bool)
            out = self.decoder(temp, e_src, tgt_padding_mask) # this should be size (B, target_size, tgt_vocab_size) note target_size starts at 1 and goes up
            # print(out.shape)
            out_next = out.argmax(-1) # index of max value. Index should contain the information we want
            # print(out_next)
            # print(out[0][0][out_next])
            target = torch.cat((target, out_next), dim=1)
            # so we only break if all batches hit end index
            # otherwise end index should still be there so in translating we cut off everything after this
            if torch.all(out_next == self.EOS_IDX):
              break

        return target # final result should be (B, seq_length, 1)
    
    def translate(self, src_sentence):
        self.eval()
        src = self.text_transform[self.src_language](src_sentence).view(1, -1)
        print(src.shape)
        num_tokens = src.shape[1]
        tgt_tokens = self.decode(src, max_len=num_tokens + 5).flatten()
        # the below translates each number into the the token it represents
        # it removes the <bos> symbol and removes the <eos> symbol
        # I use .split("<eos>") because in our batch translation we may still add tokens after <eos>
        return " ".join(self.vocab_transform[self.tgt_language].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").split("<eos>")[0]
