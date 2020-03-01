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

# constants

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
    use_full_attn = False, # set this to true for comparison with full attention
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

# optimizer

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training

def get_batch_loss(model, data):
    x, y = data
    pred = model(x)
    return F.cross_entropy(pred.transpose(1, 2), y, reduction='mean')

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_loader)
        loss = get_batch_loss(model, batch)
        loss.backward()

    print(f'training loss: {loss.item()}')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()
    
# now to evaluate

def process_inputs_chunk(fn, *args, chunks=1, dim=0):
    chunked_inputs = list(map(lambda x: x.chunk(chunks, dim=dim), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

def partial_forward(model, batch, use_full_attn=False):
    '''
    partial forward pass that returns the softmax from
    the first attention module
    '''
    x, y = batch
    t = torch.arange(x.shape[1], device=x.device)
    x = model.token_emb(x) # word embedding
    x = x + model.pos_emb(t).type(x.type()) # positional encoding
    x = model.to_model_dim(x) # linear layer

    # attention only performed on x2
    attention = model.reformer.layer_modules[0].fn
    # now do a partial run of the attention module's forward function
    device = x.device
    b, t, e, h, m = *x.shape, attention.heads, attention.num_mem_kv
    mem = attention.mem_kv.expand(b, m, e)
    keys = torch.empty(b, 0, e, dtype=mem.dtype, device=device)
    kv_len = t + m + keys.shape[1]

    x = torch.cat((x, mem, keys), dim=1)
    qk = attention.toqk(x)
    v = attention.tov(x)

    def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)

    def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

    qk = merge_heads(qk)
    v = merge_heads(v)
    attn_fn = attention.full_attn if use_full_attn else attention.lsh_attn
    partial_attn_fn = partial(attn_fn, query_len = t, input_mask = None)
    out, attn, buckets = process_inputs_chunk(partial_attn_fn, qk, v, chunks=attention.attn_chunks)
    # attn is the softmax we're looking for
    return attn

model.reformer.layer_modules[0].fn.lsh_attn._return_attn = True
attn_diff_loss = torch.zeros(NUM_BATCHES)
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='eval'):
    model.eval()

    with torch.no_grad():
        batch = next(val_loader)
        lsh_attn = partial_forward(model, batch)
        full_attn = partial_forward(model, batch, use_full_attn=True)
        loss = F.mse_loss(lsh_attn, full_attn)
        attn_diff_loss[i] = loss
mean = (sum(attn_diff_loss) / len(attn_diff_loss)).item()
print('mean = ', mean)
