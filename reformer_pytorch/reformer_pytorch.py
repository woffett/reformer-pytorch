import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from functools import partial
from itertools import chain
from revtorch import ReversibleBlock, ReversibleSequence
from pytorch_memlab import profile, MemReporter

#constants
from math import pi
TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

# helper fns

def identity(x):
    return x

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def process_inputs_chunk(fn, *args, chunks=1, dim=0):
    chunked_inputs = list(map(lambda x: x.chunk(chunks, dim=dim), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return tuple(map(lambda x: torch.cat(x, dim=dim), zip(*outputs)))

def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tensor.reshape(-1, last_dim)
    summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
    return torch.cat(summed_tensors, dim=0).reshape(orig_size)

def cache_fn(f):
    cache = None
    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def default(val, default_val):
    return default_val if val is None else val

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

# helper classes

class FixedPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0, dim, 2) / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb[None, :, :]

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, requires_grad=True))
        self.eps = eps

    def forward(self, x):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x / n * self.g

class WithNorm(nn.Module):
    def __init__(self, norm_class, dim, fn):
        super().__init__()
        self.norm = norm_class(dim)
        self.fn = fn
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x):
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c) for c in chunks], dim = self.dim)

class SettableArgs(nn.Module):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.fn = fn

    def set_args(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return self.fn(x, *self.args, **self.kwargs)

# LSH attention as described in https://openreview.net/pdf?id=rkgNKkHtvB
# adapted from trax, stripped to what paper said needed to work
# namely that buckets need to be at least 64 with 8 rounds of hashing
# https://github.com/google/trax/blob/master/trax/layers/research/efficient_attention.py#L442

def print_grad(x, printgrad = False, name=''):
    if printgrad:
        if x.grad is None:
            print('Parameter ' + name + ' has no gradient!')
        else:
            print('Parameter ' + name + ' has gradient %.3f' %
                  torch.norm(x.grad).item())

def print_module_grad(m, printgrad = False, name=''):
    if printgrad:
        [print_grad(p, name=n + ':' + n, printgrad=True) for n,p in m.named_parameters()]

def print_norm(x, printnorm = False, name=''):
    if printnorm:
        print('Parameter ' + name + ' has norm %.3f' %
              torch.norm(x).item())
        
def print_module_norm(m, printnorm = False, name=''):
    if printnorm:
        [print_norm(p, name=n + ':' + n, printnorm=True) for n,p in m.named_parameters()]

class LSHAttention(nn.Module):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  return_attn = False):
        super().__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.dropout_for_hash = nn.Dropout(drop_for_hash_rate)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

    def hash_vectors(self, n_buckets, vecs, rotations = None):
        batch_size = vecs.shape[0]
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        if rotations is None:
            random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)
        else:
            random_rotations = rotations

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)
        # a = dropped_vecs.unsqueeze(1) # insert an extra dimension at index 1
        # b = random_rotations.transpose(1,2) # bfhi -> bhfi
        # rotated_vecs = torch.matmul(a,b) # (b,1,t,f) @ (b,h,f,i) -> (b,h,t,i)

        if self._rehash_each_round:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
            # buckets is now (self.n_hashes, seqlen). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = torch.arange(self.n_hashes, device=device)
            offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
            buckets = torch.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 0)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, dim=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape 
            buckets = torch.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    def forward(self, qk, v, query_len = None, input_mask = None, rotations = None, printgrad = False):
        batch_size, seqlen, dim = qk.shape
        query_len = default(query_len, seqlen)
        device = qk.device

        n_buckets = seqlen // self.bucket_size

        buckets = self.hash_vectors(n_buckets, qk, rotations = rotations)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seqlen

        ticker = torch.arange(self.n_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # buckets_and_t = Variable(buckets_and_t, requires_grad = True)
        # buckets_and_t.register_hook(lambda g: print('grad norm: %f' % torch.norm(g).item()))

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sort_key_val(sticker, ticker, dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()
        undo_sort = undo_sort.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = self.n_hashes * n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type(bq.type())

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = max_neg_value(dots)

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), 'constant', True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        # Causal masking
        if self.causal:            
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :].clamp(max=query_len - 1)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]        
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, self.n_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, self.n_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.        
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type(dots.type())        
        dropped_dots = self.dropout(dots)        
        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(Function):
            @staticmethod
            def forward(ctx, so, slogits):
                so = so.detach()
                slogits = slogits.detach()
                o = batched_index_select(so, undo_sort)
                _, logits = sort_key_val(sticker, slogits, dim=-1)
                return o, logits

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                so_grad = batched_index_select(grad_x, sticker)
                _, slogits_grad = sort_key_val(buckets_and_t, grad_y, dim=-1)
                return so_grad, slogits_grad
            
        o, logits = UnsortLogits.apply(so, slogits)
        o = torch.reshape(o, (batch_size, self.n_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, self.n_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]
            
        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * self.n_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * self.n_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, self.n_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)
        
        # return output, attention matrix, and bucket distribution
        return out, attn, buckets

class TripletLSHAttention(LSHAttention):
    def __init__(self,
                 alpha = 1.0, # triplet loss margin
                 dim = 512, # embedding dimension
                 seq_len = 1024,
                 heads = 8, # attention heads
                 dropout = 0.,
                 bucket_size = 64,
                 n_hashes = 8,
                 causal = False,
                 allow_duplicate_attention = True,
                 attend_across_buckets = True,
                 rehash_each_round = True,
                 drop_for_hash_rate = 0.0,
                 random_rotations_per_head = False,
                 return_attn = False,
                 triplet_chunks = None
    ):
        super().__init__(dropout = dropout,
                         bucket_size = bucket_size,
                         n_hashes = n_hashes,
                         causal = causal,
                         allow_duplicate_attention = allow_duplicate_attention,
                         attend_across_buckets = attend_across_buckets,
                         rehash_each_round = rehash_each_round,
                         drop_for_hash_rate = drop_for_hash_rate,
                         random_rotations_per_head = random_rotations_per_head,
                         return_attn = return_attn)
        self.alpha = alpha
        self.seq_len = seq_len
        self.heads = heads
        n_buckets = seq_len // bucket_size
        buckets_dim = n_buckets // 2
        if self._rehash_each_round:
            buckets_dim *= n_hashes
        self.rotations = nn.Linear(dim // self.heads, buckets_dim)
        # number of chunks to split up computation of pos/neg examples for triplet loss
        self.triplet_chunks = default(triplet_chunks, dim)

    def extract_rotations(self, batch_size):
        n_buckets = self.seq_len // self.bucket_size
        rotations = self.rotations.weight.t().detach() # dim x (buckets * n_hashes / 2)
        rotations = torch.reshape(rotations, (-1, self.n_hashes, n_buckets // 2))
        rotations = rotations.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return rotations

    def triplet_examples(self, qk):
        '''
        Given a query vector, extract the highest/lower inner-product
        vectors from its LSH-sampled self-attention matrix
        '''
        batch_size, seqlen, dim = qk.shape
        device = qk.device
        rotations = self.extract_rotations(batch_size)
        # compute the rotated vecs from hash_vectors
        # | - there's repeated code here, I will clean it up later
        # | - pretty much the same as LSHAttention forward, except with max/min idxs
        
        ### BEGIN REPEATED CODE ###
        dropped_vecs = self.dropout_for_hash(qk)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, rotations)
        max_idxs = torch.argmax(rotated_vecs, dim=-1) 
        min_idxs = torch.argmin(rotated_vecs, dim=-1)

        n_buckets = seqlen // self.bucket_size
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * n_buckets, (1, -1, 1))
        max_idxs = max_idxs + offsets
        min_idxs = min_idxs + offsets
        buckets = torch.reshape(max_idxs, (batch_size, -1))
        ticker = torch.arange(self.n_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = (seqlen * buckets + (ticker % seqlen)).detach()
        # sorted according to bucket id
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        del ticker

        # detaching
        sbuckets_and_t = sbuckets_and_t.detach()
        sticker = sticker.detach()

        st = (sticker % seqlen)
        sqk = batched_index_select(qk, st)

        # split off the bin axis
        chunk_size = self.n_hashes * n_buckets
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)
        
        bk = look_one_back(bqk)
        del sqk
        del bqk
        ### END REPEATED CODE ###
        # now, bk has shape (batch x C x S x dim)
        #  - C = n_hashes * n_buckets, the number of chunks to attend over
        #  - S = 2 * seqlen / n_buckets, chunks concat'd w/ prev. chunk

        # reshape min/max idxs to batch x (seqlen*n_hashes)
        max_idxs = torch.reshape(torch.transpose(max_idxs,-1,-2),
                                 (batch_size, -1))
        min_idxs = torch.reshape(torch.transpose(min_idxs,-1,-2),
                                 (batch_size, -1))
        
        # batch index select over last two dimensions
        def batched_index_select2(values, indices):
            last_dim = values.shape[-1]
            snd_last_dim = values.shape[-2]
            return values.gather(
                1,
                indices[:,:,None,None].expand(-1,-1,snd_last_dim, last_dim)
            )

        def pos_neg_examples(min_idxs, max_idxs, qk, bk):
            max_batches = batched_index_select2(bk, max_idxs)
            min_batches = batched_index_select2(bk, min_idxs)
            # reshape to make dot product easier
            max_batches = max_batches.reshape(batch_size, -1, self.n_hashes,
                                              seqlen // n_buckets * 2, dim)
            min_batches = min_batches.reshape(batch_size, -1, self.n_hashes,
                                              seqlen // n_buckets * 2, dim)

            # perform dot product between qk and the vectors it attends over
            max_self_attn = torch.einsum('bthkd,btdz->bthkz', max_batches,
                                         qk.unsqueeze(-1)).squeeze(-1)
            min_self_attn = torch.einsum('bthkd,btdz->bthkz', min_batches,
                                         qk.unsqueeze(-1)).squeeze(-1)
            
            # find the indices of the maximum dot prod. across all n_hashes
            max_self_attn_idxs = torch.argmax(
                max_self_attn.reshape(batch_size, -1, self.n_hashes * seqlen // n_buckets * 2),
                dim=-1).unsqueeze(-1)        
            min_self_attn_idxs = torch.argmin(
                min_self_attn.reshape(batch_size, -1, self.n_hashes * seqlen // n_buckets * 2),
                dim=-1).unsqueeze(-1)

            # select along second last dim
            min_batches = min_batches.reshape(batch_size, -1,
                                              self.n_hashes * seqlen // n_buckets * 2, dim)
            max_batches = max_batches.reshape(batch_size, -1,
                                              self.n_hashes * seqlen // n_buckets * 2, dim)
            pos_vectors = max_batches.gather(
            2,
            max_self_attn_idxs[:,:,:,None].expand(-1,-1, -1, dim)
            ).squeeze(-2)
            neg_vectors = min_batches.gather(
                2,
                min_self_attn_idxs[:,:,:,None].expand(-1,-1, -1, dim)
            ).squeeze(-2)
            # pos/neg_vectors have same shape as qk            
            return pos_vectors.detach(), neg_vectors.detach()

        partial_select_fn = partial(pos_neg_examples, bk = bk)
        pos_vectors, neg_vectors = process_inputs_chunk(
            partial_select_fn, min_idxs, max_idxs, qk, chunks=self.triplet_chunks, dim=1
        )
        return pos_vectors, neg_vectors

    def triplet_forward(self,
                        x, # input
                        p, # positive example
                        n, # negative example
    ):
        '''
        Given inputs, positive and negative examples, compute the 
        triplet loss given by cosine similarity
        '''
        emb_x = self.rotations(x)
        emb_p = self.rotations(p)
        emb_n = self.rotations(n)

        # cosine similarity
        sim_xp = F.cosine_similarity(emb_x, emb_p, dim=-1)
        sim_xn = F.cosine_similarity(emb_x, emb_n, dim=-1)

        # distance in radians
        # dis_xp = 1 - torch.acos(sim_xp)/pi
        # dis_xn = 1 - torch.acos(sim_xn)/pi
        dis_xp = 1 - sim_xp
        dis_xn = 1 - sim_xn

        triplet_loss = dis_xp - dis_xn + self.alpha
        triplet_loss = torch.mean(torch.max(triplet_loss,
                                            torch.zeros(triplet_loss.size()).to(x.device)))
        return triplet_loss

    def forward(self, qk, v, query_len = None, input_mask = None, printgrad = False):
        batch_size, seqlen, dim = qk.shape
        n_buckets = self.seq_len // self.bucket_size
        rotations = self.extract_rotations(batch_size)
        out, attn, buckets = super().forward(qk, v,
                                             query_len = query_len,
                                             input_mask = input_mask,
                                             rotations = rotations,
                                             printgrad = printgrad)
            
        return out, attn, buckets

# simple full attention

class FullQKAttention(nn.Module):
    def __init__(self, causal = False):
        super().__init__()
        self.causal = causal

    def forward(self, qk, v, query_len = None, input_mask = None, **kwargs):
        b, seq_len, dim = qk.shape
        query_len = default(query_len, seq_len)
        t = query_len

        q = qk[:, 0:query_len]
        qk = F.normalize(qk, 2, dim=-1).type(q.type())

        dot = torch.einsum('bie,bje->bij', q, qk) * (dim ** -0.5)

        # qk attention requires tokens not attend to self
        i = torch.arange(t)
        dot[:, i, i] = TOKEN_SELF_ATTN_VALUE
        masked_value = max_neg_value(dot)        

        # Input mask for padding in variable lengthed sequences
        if input_mask is not None:
            mask = input_mask[:, :, None] * input_mask[:, None, :]
            mask = F.pad(mask, (0, seq_len - mask.shape[-1]), 'constant', True)
            dot.masked_fill_(~mask, masked_value)
            
        if self.causal:
            i, j = torch.triu_indices(t, t, 1)
            dot[:, i, j] = masked_value
        
        dot = dot.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dot, v)

        return out, dot, torch.empty(0)

# Shared qk attention, using either full or LSH attention

class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, bucket_size = 64, n_hashes = 8, causal = False, attn_chunks = None, random_rotations_per_head = False, attend_across_buckets = True, allow_duplicate_attention = True, num_mem_kv = 0, attn_type = 'lsh', full_attn_thres = None, return_attn = False, max_seq_len = None, batch = None, alpha = 1.0, triplet_chunks = None, **kwargs):
        super().__init__()
        assert dim % heads == 0, 'dimensions must be divisible by number of heads'

        self.dim = dim
        self.heads = heads
        self.attn_chunks = default(attn_chunks, heads)

        self.toqk = nn.Linear(dim, dim, bias = False)
        self.tov = nn.Linear(dim, dim, bias = False)
        self.to_out = nn.Linear(dim, dim)

        self.bucket_size = bucket_size
        self.attn_type = attn_type
        if self.attn_type == 'triplet':
            self.lsh_attn = TripletLSHAttention(alpha = alpha, dim = self.dim, seq_len = max_seq_len, heads = self.heads, bucket_size = bucket_size, n_hashes = n_hashes, causal = causal, random_rotations_per_head = random_rotations_per_head, attend_across_buckets = attend_across_buckets, allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn, triplet_chunks = triplet_chunks, **kwargs)
        else:
            self.lsh_attn = LSHAttention(bucket_size=bucket_size, n_hashes=n_hashes, causal=causal, random_rotations_per_head=random_rotations_per_head, attend_across_buckets = attend_across_buckets,  allow_duplicate_attention = allow_duplicate_attention, return_attn = return_attn, **kwargs)
        
        self.full_attn = FullQKAttention(causal = causal)
        self.full_attn_thres = default(full_attn_thres, bucket_size)

        self.num_mem_kv = num_mem_kv
        self.mem_kv = nn.Parameter(torch.randn(1, num_mem_kv, dim, requires_grad=True))

        self.callback = None
        self.triplet_loss = None

    def forward(self, x, keys = None, input_mask = None, calc_triplet = False):
        device = x.device
        b, t, e, h, m = *x.shape, self.heads, self.num_mem_kv

        mem = self.mem_kv.expand(b, m, e)
        keys = default(keys, torch.empty(b, 0, e, dtype=mem.dtype, device=device))

        kv_len = t + m + keys.shape[1]
        use_full_attn = self.attn_type == 'full' or kv_len <= self.full_attn_thres

        if not use_full_attn:
            assert not use_full_attn and (kv_len % self.bucket_size == 0), f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}'

        x = torch.cat((x, mem, keys), dim=1)
        qk = self.toqk(x)
        v = self.tov(x)

        def merge_heads(v):
            return v.view(b, kv_len, h, -1).transpose(1, 2).reshape(b * h, kv_len, -1)

        def split_heads(v):
            return v.view(b, h, t, -1).transpose(1, 2).contiguous()

        qk = merge_heads(qk)
        v = merge_heads(v)

        attn_fn = self.full_attn if use_full_attn else self.lsh_attn
        partial_attn_fn = partial(attn_fn, query_len = t, input_mask = input_mask)
        out, attn, buckets = process_inputs_chunk(partial_attn_fn,
                                                  qk, v, chunks=self.attn_chunks)
        out = split_heads(out).view(b, t, e)

        if self.callback is not None:
            self.callback(attn.reshape(b, h, t, -1), buckets.reshape(b, h, -1))

        if self.attn_type == 'triplet' and calc_triplet:
            pos_vectors, neg_vectors = process_inputs_chunk(
                self.lsh_attn.triplet_examples,
                qk, chunks=self.attn_chunks)
            
            def chunked_loss(fn, *args, chunks=1, dim=0):
                chunked_inputs = list(map(lambda x: x.chunk(chunks, dim=dim), args))
                outputs = [fn(*inputs) for inputs in zip(*chunked_inputs)]
                return sum(outputs)
            
            triplet_loss = chunked_loss(self.lsh_attn.triplet_forward,
                                        qk, pos_vectors, neg_vectors,
                                        chunks=self.attn_chunks, dim=1)
            if self.triplet_loss is None:
                self.triplet_loss = triplet_loss
            else:
                self.triplet_loss += triplet_loss

        return self.to_out(out)

# feed forward

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            GELU(),
            nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)

# reformer lm

class Reformer(nn.Module):
    def __init__(self, dim, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., lsh_attend_across_buckets = True, lsh_allow_duplicate_attention = True, random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, attn_type = 'lsh', full_attn_thres = None, num_mem_kv = 0, batch = None, triplet_chunks = None):
        super().__init__()
        self.dim = dim
        self.depth = depth

        get_attn = lambda: SettableArgs(LSHSelfAttention(dim, heads, bucket_size, n_hashes, causal = causal, dropout = lsh_dropout, attn_chunks = attn_chunks, allow_duplicate_attention = lsh_allow_duplicate_attention, attend_across_buckets = lsh_attend_across_buckets, random_rotations_per_head = random_rotations_per_head, num_mem_kv = num_mem_kv, attn_type = attn_type, full_attn_thres = full_attn_thres, max_seq_len = max_seq_len, batch = batch, triplet_chunks = triplet_chunks))
        get_ff = lambda: FeedForward(dim)

        if weight_tie:
            get_attn = cache_fn(get_attn)
            get_ff = cache_fn(get_ff)

        blocks = []
        norm_type = ScaleNorm if use_scale_norm else nn.LayerNorm

        for _ in range(depth):
            attn = get_attn()
            parallel_net = get_attn() if twin_attention else get_ff()

            f = WithNorm(norm_type, dim, attn)
            g = WithNorm(norm_type, dim, parallel_net)

            if not twin_attention and ff_chunks > 1:
                g = Chunk(ff_chunks, g, along_dim = -2)

            blocks.append(ReversibleBlock(f, g, split_along_dim=-1, fix_random_seed=True))

        self.layers = ReversibleSequence(nn.ModuleList(blocks), eagerly_discard_variables = False)
        self.layer_modules = list(chain(*[[m.f_block.fn, m.g_block.fn] for m in blocks]))

    def set_reversible_args(self, *args, **kwargs):
        for module in self.layer_modules:
            if isinstance(module, SettableArgs):
                module.set_args(*args, **kwargs)

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim = -1)
        self.set_reversible_args(**kwargs)
        x = self.layers(x)
        return torch.stack(x.chunk(2, dim=-1)).sum(dim=0)

class ReformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads = 8, bucket_size = 64, n_hashes = 8, ff_chunks = 100, attn_chunks = None, causal = False, weight_tie = False, lsh_dropout = 0., random_rotations_per_head = False, twin_attention = False, use_scale_norm = False, attn_type = 'lsh', full_attn_thres = None, num_mem_kv = 0, emb_dim = None, return_embeddings = False, fixed_position_emb = False, batch = None, triplet_chunks = None):
        super().__init__()
        emb_dim = default(emb_dim, dim)
        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = FixedPositionEmbedding(emb_dim) if fixed_position_emb else nn.Embedding(max_seq_len, emb_dim)
        self.to_model_dim = identity if emb_dim == dim else nn.Linear(emb_dim, dim)
        self.dim = dim

        self.reformer = Reformer(dim, depth, max_seq_len, heads = heads, bucket_size = bucket_size, n_hashes = n_hashes, ff_chunks = ff_chunks, attn_chunks = attn_chunks, causal = causal, weight_tie = weight_tie, lsh_dropout = lsh_dropout, random_rotations_per_head = random_rotations_per_head, twin_attention = twin_attention, use_scale_norm = use_scale_norm, attn_type = attn_type, full_attn_thres = full_attn_thres, num_mem_kv = num_mem_kv, batch = batch, triplet_chunks = triplet_chunks)
        self.to_logits = identity if return_embeddings else nn.Linear(dim, num_tokens)

    def forward(self, x, **kwargs):
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x)
        x = x + self.pos_emb(t).type(x.type())

        x = self.to_model_dim(x)
        x = self.reformer(x, **kwargs)
        return self.to_logits(x)

    def clear_non_rotation_gradients(self):
        # clear gradients from triplet loss
        for i in range(len(self.reformer.layer_modules)//2):
            f = self.reformer.layer_modules[2*i].fn
            g = self.reformer.layer_modules[(2*i)+1].fn
            # only zero out toqk, tov, and to_out
            # leave gradients of rotations
            f.toqk.zero_grad()
            f.tov.zero_grad()
            f.to_out.zero_grad()
            g.zero_grad()

    def get_triplet_loss(self):
        total = 0
        for i in range(len(self.reformer.layer_modules)//2):
            f = self.reformer.layer_modules[2*i].fn
            if f.triplet_loss is not None:
                total += f.triplet_loss
        return total

    def clear_triplet_loss(self):
        for i in range(len(self.reformer.layer_modules)//2):
            f = self.reformer.layer_modules[2*i].fn
            if f.triplet_loss is not None:
                f.triplet_loss = None
