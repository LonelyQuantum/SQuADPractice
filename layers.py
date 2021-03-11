"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

import sys

class Embedding_CW(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, char_vectors, char_cnn_output_size, hidden_size, drop_prob):
        super(Embedding_CW, self).__init__()
        self.drop_prob = drop_prob
        self.char_emb_size = char_vectors.size(1)
        self.char_cnn_output_size = char_cnn_output_size
        self.char_cnn_kernel_size = 5
        self.char_limit = 16
        self.embed_w = nn.Embedding.from_pretrained(word_vectors)
        self.embed_c = nn.Embedding.from_pretrained(char_vectors)
        self.cnn_c = nn.Conv1d(self.char_emb_size, self.char_cnn_output_size, self.char_cnn_kernel_size)
        self.maxpool_c = nn.MaxPool1d(self.char_limit - self.char_cnn_kernel_size + 1)
        self.proj = nn.Linear(word_vectors.size(1)+self.char_cnn_output_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idxs, c_idxs):
        emb_w = self.embed_w(w_idxs)   # (batch_size, seq_len, embed_size)
        emb_c = self.embed_c(c_idxs)   # (batch_size, seq_len, word_len, embed_size)
        emb_c = emb_c.view([-1,emb_c.size(2),emb_c.size(3)]).permute([0,2,1])
        emb_c = self.cnn_c(emb_c)
        emb_c = self.maxpool_c(emb_c).squeeze().view([emb_w.size(0),emb_w.size(1),self.char_cnn_output_size])
        emb = torch.cat([emb_w,emb_c],dim=2)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb

class Embedding_W(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding_W, self).__init__()
        self.drop_prob = drop_prob
        self.embed_w = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, w_idxs):
        emb = self.embed_w(w_idxs)   # (batch_size, seq_len, embed_size)    
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb

class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Flatten parameters
        self.rnn.flatten_parameters()

        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True).cuda()

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s

class CoAttention(nn.Module):
    """Coattention originally used by DCN.

    Coattention layer adds a second level attention of the first level C2Q and Q2C 
    attentions. The output would be the BiLSTM hidden states of the concatenation 
    of the first level C2Q attention and the second level attention.

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(CoAttention, self).__init__()
        self.drop_prob = drop_prob
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.c_cent = nn.Parameter(torch.zeros(1, hidden_size))
        self.q_cent = nn.Parameter(torch.zeros(1, hidden_size))
        self.rnn = RNNEncoder(input_size=hidden_size*2,
                              hidden_size=hidden_size*2, 
                              num_layers=1, 
                              drop_prob=drop_prob)
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        for weight in (self.c_cent, self.q_cent):
            nn.init.xavier_uniform_(weight)

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        # Get the projected q
        q = self.proj(q)
        q = F.dropout(q, self.drop_prob, self.training)
        q = F.relu(q) # (batch_size, q_len, hidden_size)
        # Add sentinel vectors
        q = torch.cat([q,self.q_cent.unsqueeze(0).repeat(batch_size,1,1)], dim=1) # (batch_size, q_len+1, hidden_size)
        c = torch.cat([c,self.c_cent.unsqueeze(0).repeat(batch_size,1,1)], dim=1) # (batch_size, c_len+1, hidden_size)
        q_mask = torch.cat([q_mask, torch.ones(batch_size, 1).to(self.device)], dim=1) # (batch_size, q_len+1)
        c_mask = torch.cat([c_mask, torch.ones(batch_size, 1).to(self.device)], dim=1) # (batch_size, c_len+1)
        # Compute affinity matrix
        L = torch.bmm(c, q.transpose(1,2)) # (bs, c_len+1, hid_size) x (bs, hid_size, q_len+1) => (bs, c_len+1, q_len+1)
        # Compute C2Q attention
        alpha = masked_softmax(L, q_mask.view(batch_size, 1, q_len+1), dim=2)  # (batch_size, c_len+1, q_len+1)
        c2q_att = torch.bmm(alpha, q)  # (bs, c_len+1, q_len+1) x (bs, q_len+1, hid_size) => (bs, c_len+1, hid_size)
        # Compute Q2C attention
        beta = masked_softmax(L, c_mask.view(batch_size, c_len+1, 1), dim=1) # (batch_size, c_len+1, q_len+1)
        q2c_att = torch.bmm(beta.transpose(1,2), c) # (bs, q_len+1, c_len+1) x (bs, c_len+1, hid_size) => (bs, q_len+1, hid_size)
        # Compute second level attention
        att_lv2 = torch.bmm(alpha, q2c_att)  # (bs, c_len+1, q_len+1) x (bs, q_len+1, hid_size) => (bs, c_len+1, hid_size)
        x = torch.cat([att_lv2, c2q_att], dim=2) # (batch_size, c_len+1, hidden_size*2)
        x = self.rnn(x[:,:c_len,:], c_mask[:,:c_len].sum(-1)) # (batch_size, c_len, hidden_size*4)
        return x


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
