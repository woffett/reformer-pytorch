'''
A sandbox for playing around with models
'''

from reformer_pytorch import ReformerLM

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
from math import pi

NUM_BATCHES = 10
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 5
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
# SEQ_LEN = 4096
SEQ_LEN = 4096
SEED = 1

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# set random seeds
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)
            
def process_inputs_chunk(fn, *args, chunks=1, dim=0):
    chunked_inputs = list(map(lambda x: x.chunk(chunks, dim=dim), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def look_one_back(x):
    x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
    return torch.cat([x, x_extra], dim=2)

def get_top_p(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    return logits

def sample_next_token(logits, top_p=0.9, temperature = 1.0):
    logits = logits[0, -1, :] / temperature
    filtered_logits = get_top_p(logits, top_p=top_p)

    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, 1)

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))
# instantiate model

model = ReformerLM(
    dim = 512,
    depth = 6,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 64,
    n_hashes = 4,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = False, # need to set this to false!
    causal = True,
    # use_full_attn = False, # set this to true for comparison with full attention
    attn_type = 'simhash',
    store_stats = True,
    batch = BATCH_SIZE
)
model.cuda()
# model.load_state_dict(torch.load('../enwik8_simple/4096_learned_64bsize_4rounds.pt'))

with gzip.open('../enwik8_simple/data/enwik8.gz') as file:
    X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
    trX, vaX = np.split(X, [int(90e6)])
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
        return full_seq[0:-1].cuda(), full_seq[1:].cuda()

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset   = TextSamplerDataset(data_val, SEQ_LEN)
train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

def get_batch_loss(model, data):
    x, y = data
    pred = model(x)
    return F.cross_entropy(pred.transpose(1, 2), y, reduction='mean')

# model.reformer.layer_modules[0].fn.lsh_attn._return_attn = True
batch = next(train_loader)
x, y = batch

# BELOW: testing code for figuring out how to calculate pos/neg examples for triplet loss
# t = torch.arange(x.shape[1], device=x.device)

# # partial activation z
# z = model.token_emb(x)
# z = z + model.pos_emb(t).type(z.type())
# z = model.to_model_dim(z)

# target_layer = 1

# for i in range(0,target_layer*2,2):
#     z = F.layer_norm(z, (model.dim,))
#     zprime = model.reformer.layer_modules[i].fn(z)
#     z = zprime + z

# f = model.reformer.layer_modules[2 * target_layer].fn

    
# b, t, e = z.shape
# # f = model.reformer.layer_modules[6].fn
# # g = model.reformer.layer_modules[7].fn

# z = F.layer_norm(z, (model.dim,))
# # y2 = g(z)
# h = f.heads
# qk = f.toqk(z)
# v = f.tov(z)
# kv_len = t

# def merge_heads(v):
#     return v.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)

# def split_heads(v):
#     return v.view(b, h, t, -1).transpose(1, 2).contiguous()
            
# qk = merge_heads(qk)
# v = merge_heads(v)
# attn_fn = f.lsh_attn
# # prev_return_setting = attn_fn._return_attn
# # attn_fn._return_attn = True
# # partial_attn_fn = partial(attn_fn, query_len = t, input_mask = None)
# # with torch.no_grad():
# #     out, attn, _ = process_inputs_chunk(partial_attn_fn,
# #                                         qk, v, chunks=f.attn_chunks)

# # metrics: avg. (normalized) inner product within buckets

# qk_chunks = qk.chunk(f.attn_chunks, dim=0)
# v_chunks = v.chunk(f.attn_chunks, dim=0)
# qk = qk_chunks[1]
# v = v_chunks[1]

# batch_size, seqlen, dim = qk.shape
# query_len = seqlen
# device = qk.device
# n_buckets = seqlen // attn_fn.bucket_size
# buckets = attn_fn.hash_vectors(n_buckets, qk, rotations = None)
# max_idxs = buckets.reshape(batch_size, -1, seqlen)

# ticker = torch.arange(attn_fn.n_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
# buckets_and_t = seqlen * buckets + (ticker % seqlen)
# buckets_and_t = buckets_and_t.detach()

# sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
# _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
# del ticker

# st = (sticker % seqlen)
# sqk = batched_index_select(qk, st)

# chunk_size = attn_fn.n_hashes * n_buckets
# bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))

# def look_one_back(x):
#     x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
#     return torch.cat([x, x_extra], dim=2)

# bk = look_one_back(bqk)

# # reshape min/max idxs to batch x (seqlen*n_hashes)
# max_idxs = torch.reshape(torch.transpose(max_idxs,-1,-2),
#                          (batch_size, -1))

# def batched_index_select2(values, indices):
#     last_dim = values.shape[-1]
#     snd_last_dim = values.shape[-2]
#     return values.gather(
#         1,
#         indices[:,:,None,None].expand(-1,-1,snd_last_dim, last_dim)
#     )

# def pos_neg_examples(max_idxs, qk, bk):
#     max_batches = batched_index_select2(bk, max_idxs)
    
#     # reshape to make dot product easier
#     max_batches = max_batches.reshape(batch_size, -1, self.n_hashes,
#                                       seqlen // n_buckets * 2, dim)

#     # perform dot product between qk and the vectors it attends over
#     max_self_attn = torch.einsum('bthkd,btdz->bthkz', max_batches,
#                                  qk.unsqueeze(-1)).squeeze(-1)
            
#     # find the indices of the maximum dot prod. across all n_hashes
#     max_self_attn_idxs = torch.argmax(
#         max_self_attn.reshape(batch_size, -1, self.n_hashes * seqlen // n_buckets * 2),
#         dim=-1).unsqueeze(-1)        
#     min_self_attn_idxs = torch.argmin(
#         max_self_attn.reshape(batch_size, -1, self.n_hashes * seqlen // n_buckets * 2),
#         dim=-1).unsqueeze(-1)

#     # select along second last dim
#     max_batches = max_batches.reshape(batch_size, -1,
#                                       self.n_hashes * seqlen // n_buckets * 2, dim)
#     pos_vectors = max_batches.gather(
#         2,
#         max_self_attn_idxs[:,:,:,None].expand(-1,-1, -1, dim)
#     ).squeeze(-2)
#     neg_vectors = max_batches.gather(
#         2,
#         min_self_attn_idxs[:,:,:,None].expand(-1,-1, -1, dim)
#     ).squeeze(-2)
#     # pos/neg_vectors have same shape as qk            
#     return pos_vectors.detach(), neg_vectors.detach()

# partial_select_fn = partial(pos_neg_examples, bk = bk)
# pos_vectors, neg_vectors = process_inputs_chunk(
#     partial_select_fn, max_idxs, qk, chunks=self.triplet_chunks, dim=1
# )

# dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)

# mean_inner_prods = torch.mean(dots, dim=-1)
# std_inner_prods = dots.std(dim=-1)
# var_inner_prods = std_inner_prods ** 2
# print('Inner prod mean, var = (%.3f, %.3f)' % (torch.mean(mean_inner_prods).item(),
#                                                torch.mean(var_inner_prods).item()))

# min_vals, min_idxs = torch.min(dots, dim=-1)
# # Mask out attention to self except when no other targets are available.
# self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
# dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
# del self_mask

# max_vals, max_idxs = torch.max(dots, dim=-1)

# avg_max_vals = torch.mean(max_vals)
# avg_min_vals = torch.mean(min_vals)

# print('Avg min, max bucket vals = (%.3f, %.3f)' % (avg_min_vals, avg_max_vals))

# attn_fn._return_attn = prev_return_setting
