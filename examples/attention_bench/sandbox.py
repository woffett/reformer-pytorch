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
SEQ_LEN = 4096
SEED = 1

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
    depth = 1,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 8,
    n_hashes = 4,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = True,
    causal = True,
    # use_full_attn = False, # set this to true for comparison with full attention
    attn_type = 'lsh',
    parameterized_rotations = True,
    batch = BATCH_SIZE
)
model.cuda()

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

model.reformer.layer_modules[0].fn.lsh_attn._return_attn = True
batch = next(train_loader)
x, y = batch
# loss = get_batch_loss(model, batch)
# loss.backward()
# triplet_loss = model.triplet_forward(x)
# triplet_loss.backward()

# BELOW: testing code for figuring out how to calculate pos/neg examples for triplet loss
t = torch.arange(x.shape[1], device=x.device)

# partial activation z
z = model.token_emb(x)
z = z + model.pos_emb(t).type(z.type())
z = model.to_model_dim(z)

b, t, e = z.shape
f = model.reformer.layer_modules[0].fn
g = model.reformer.layer_modules[1].fn

z = F.layer_norm(z, (model.dim,))
y2 = g(z)
h = f.heads
qk = f.toqk(z)
v = f.tov(z)
kv_len = t

def merge_heads(v):
    return v.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)

def split_heads(v):
    return v.view(b, h, t, -1).transpose(1, 2).contiguous()
            
qk = merge_heads(qk)
v = merge_heads(v)
attn_fn = f.triplet_lsh_attn
prev_return_setting = attn_fn._return_attn
attn_fn._return_attn = True
partial_attn_fn = partial(attn_fn, query_len = t, input_mask = None)
out, attn, _ = process_inputs_chunk(partial_attn_fn,
                                    qk, v, chunks=f.attn_chunks)

# delving into the LSH function

# first, chunk our inputs so we can fit into memory
chunked_inputs = list(map(lambda x: x.chunk(attn_fn.heads, dim=0), [qk, v]))
# just work on the first ones as an example for now
qk, v = chunked_inputs[0][0], chunked_inputs[1][0]

# generating appropriate rotations
batch_size, seqlen, dim = qk.shape
device = qk.device
n_buckets = attn_fn.seq_len // attn_fn.bucket_size
rotations = attn_fn.rotations.weight.t().detach()
rotations = torch.reshape(rotations, (-1, attn_fn.n_hashes, n_buckets // 2))
rotations = rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)

# generating intermediate tensors for bucketing
dropped_vecs = attn_fn.dropout_for_hash(qk)
rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, rotations)
max_idxs = torch.argmax(rotated_vecs, dim=-1)
min_idxs = torch.argmin(rotated_vecs, dim=-1)
offsets = torch.arange(attn_fn.n_hashes, device=device)
offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
max_idxs = max_idxs + offsets
min_idxs = min_idxs + offsets
buckets = torch.reshape(max_idxs, (batch_size, -1))

ticker = torch.arange(attn_fn.n_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
buckets_and_t = (seqlen * buckets + (ticker % seqlen)).detach()
# sorted according to bucket id
sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
_, undo_sort = sort_key_val(sticker, ticker, dim=-1)
del ticker

# detaching
sbuckets_and_t = sbuckets_and_t.detach()
sticker = sticker.detach()
undo_sort = undo_sort.detach()

st = (sticker % seqlen)
sqk = batched_index_select(qk, st)
sv = batched_index_select(v, st)

# split off the bin axis
chunk_size = attn_fn.n_hashes * n_buckets
bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
bk = look_one_back(bqk)

# reshape min/max idxs
max_idxs = torch.reshape(torch.transpose(max_idxs,-1,-2), (batch_size, -1))
min_idxs = torch.reshape(torch.transpose(min_idxs,-1,-2), (batch_size, -1))

# batch index select over last two dimensions
def batched_index_select2(values, indices):
    last_dim = values.shape[-1]
    snd_last_dim = values.shape[-2]
    return values.gather(1, indices[:,:,None,None].expand(-1,-1,snd_last_dim, last_dim))
last_dim = bk.shape[-1]
max_batches = batched_index_select2(bk, max_idxs)
min_batches = batched_index_select2(bk, min_idxs)
max_batches = max_batches.reshape(-1, seqlen, attn_fn.n_hashes, seqlen // n_buckets * 2, last_dim)
min_batches = min_batches.reshape(-1, seqlen, attn_fn.n_hashes, seqlen // n_buckets * 2, last_dim)

max_self_attn = torch.einsum('bthkd,btdz->bthkz', max_batches, qk.unsqueeze(-1)).squeeze(-1)
min_self_attn = torch.einsum('bthkd,btdz->bthkz', min_batches, qk.unsqueeze(-1)).squeeze(-1)

# take the maximum inner product over union of vectors in all n_hashes
min_self_attn_idxs = torch.argmin(min_self_attn.reshape(batch_size, seqlen, -1), dim=-1).unsqueeze(-1)
# need to use second largest in max, so as not to use yourself!
max_self_attn_idxs = torch.argmax(max_self_attn.reshape(batch_size, seqlen, -1), dim=-1).unsqueeze(-1)

# select along second last dim
min_batches = min_batches.reshape(batch_size, seqlen, -1, last_dim)
max_batches = max_batches.reshape(batch_size, seqlen, -1, last_dim)
snd_last_dim = min_batches.shape[-2]
pos_vectors = max_batches.gather(
    2,
    max_self_attn_idxs[:,:,:,None].expand(-1,-1, -1, last_dim)
).squeeze(-2).detach()
neg_vectors = min_batches.gather(
    2,
    min_self_attn_idxs[:,:,:,None].expand(-1,-1, -1, last_dim)
).squeeze(-2).detach()

# if we look at cosine distance, we can confirm that pos_vectors get a higher cosine sim

emb_x = attn_fn.rotations(qk)
emb_p = attn_fn.rotations(pos_vectors)
emb_n = attn_fn.rotations(neg_vectors)

sim_xp = F.cosine_similarity(emb_x, emb_p, dim=-1)
sim_xn = F.cosine_similarity(emb_x, emb_n, dim=-1)

dis_xp = 1 - sim_xp
dis_xn = 1 - sim_xn
triplet = dis_xp - dis_xn + attn_fn.alpha
triplet = torch.mean(torch.max(triplet,
                               torch.zeros(triplet.size()).to(qk.device)))

attn_fn._return_attn = prev_return_setting
