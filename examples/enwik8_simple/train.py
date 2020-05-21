from reformer_pytorch import ReformerLM

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# constants

# NUM_BATCHES = int(1e5)
NUM_BATCHES = 100
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
TRIPLET_LEARNING_RATE = 0.00075
# VALIDATE_EVERY  = 100
VALIDATE_EVERY  = 10
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
SEQ_LEN = 4096
ATTN_TYPE = 'lsh'
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
    depth = 6,
    max_seq_len = SEQ_LEN,
    num_tokens = 256,
    heads = 8,
    bucket_size = 64,
    n_hashes = 4,
    ff_chunks = 10,
    lsh_dropout = 0.1,
    weight_tie = False,
    causal = True,
    # use_full_attn = False # set this to true for comparison with full attention
    attn_type = ATTN_TYPE,
    store_stats = True,
    triplet_chunks = 512
)

model.cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
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
triplet_optim = torch.optim.Adam(model.parameters(), lr=TRIPLET_LEARNING_RATE)

# training

def get_batch_loss(model, data):
    x, y = data
    pred = model(x, calc_triplet = True)
    return F.cross_entropy(pred.transpose(1, 2), y, reduction='mean')

# generating Tensorboard writer
writer = SummaryWriter()

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(train_loader)
        loss = get_batch_loss(model, batch)
        loss.backward()

    print(f'training loss: {loss.item()}')
    writer.add_scalar('Loss/train', loss, i)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    # now for triplet loss
    if ATTN_TYPE == 'triplet':
        triplet_loss = model.get_triplet_loss()
        # for logging, show mean loss
        print(f'training triplet loss: {triplet_loss.item()/GRADIENT_ACCUMULATE_EVERY}')
        writer.add_scalar('Loss/train_triplet', triplet_loss/GRADIENT_ACCUMULATE_EVERY, i)
        triplet_loss.backward()
        model.clear_non_rotation_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        triplet_optim.step()
        triplet_optim.zero_grad()
        model.clear_triplet_loss()

    means, variances, noninfs = model.get_statistics(GRADIENT_ACCUMULATE_EVERY)
    for j, (med, vari, noninf) in enumerate(zip(means, variances, noninfs)):
        writer.add_scalar('Mean/train/%d' % j, med, i)
        writer.add_scalar('Variance/train/%d' % j, vari, i)
        writer.add_scalar('Noninfs/train/%d' % j, noninf, i)
    
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            batch = next(val_loader)
            loss = get_batch_loss(model, batch)
            writer.add_scalar('Loss/val', loss, i)
            print(f'validation loss: {loss.item()}')            
            if ATTN_TYPE == 'triplet':
                triplet = model.get_triplet_loss()
                writer.add_scalar('Loss/val_triplet', triplet_loss, i)
                print(f'validation triplet loss: {triplet_loss.item()}')
                model.clear_triplet_loss()

            means, variances, noninfs = model.get_statistics(BATCH_SIZE)
            for j, (med, vari, noninf) in enumerate(zip(means, variances, noninfs)):
                writer.add_scalar('Mean/val/%d' % j, med, i)
                writer.add_scalar('Variance/val/%d' % j, vari, i)
                writer.add_scalar('Noninfs/val/%d' % j, noninf, i)  

    # if i % GENERATE_EVERY == 0:
    #     model.eval()
    #     with torch.no_grad():
    #         inp, _ = random.choice(val_dataset)
    #         output_str = ''
    #         prime = decode_tokens(inp)

    #         print(f'%s \n\n %s', (prime, '*' * 100))

    #         for _ in tqdm.tqdm(range(GENERATE_LENGTH), desc='generating'):
    #             logits = model(inp[None, :])
    #             next_token = sample_next_token(logits)
    #             output_str += decode_token(next_token)
    #             inp = torch.cat((inp[1:], next_token), dim=0)

    #         print(output_str)

writer.close()
# torch.save(model.state_dict(), '4096_learned_64bsize_4rounds.pt')
