import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer

from vlc2flc.vlc2flc_configs import PAD_INDEX


class TargetPE(nn.Module):
    def __init__(self, emb_size: int, dropout: float, max_len: int = 500):
        super(TargetPE, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tgt_embedding):
        return self.dropout(tgt_embedding + self.pos_embedding[:tgt_embedding.size(0), :])


class SourcePE(nn.Module):
    def __init__(self, emb_size, max_width, max_height, dropout):
        super(SourcePE, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size / 4) * math.log(10000) / emb_size / 4)

        x_p = torch.arange(0, max_width).reshape(max_width, 1)
        x_pe = torch.sin(x_p * den)

        y_p = torch.arange(0, max_height).reshape(max_height, 1)
        y_pe = torch.sin(y_p * den)

        self.register_buffer('x_pe', x_pe)
        self.register_buffer('y_pe', y_pe)

        self.emb_size = emb_size

        self.dropout = nn.Dropout(dropout)

    def get_pe(self, src_boxes):
        pe = torch.zeros((len(src_boxes), self.emb_size), requires_grad=False)
        for index, src_box in enumerate(src_boxes):
            pe[index, 0::4] = self.x_pe[src_box[0]]
            pe[index, 1::4] = self.y_pe[src_box[1]]
            pe[index, 2::4] = self.x_pe[src_box[2]]
            pe[index, 3::4] = self.y_pe[src_box[3]]

    def forward(self, src_embedding, src_boxes):
        return self.dropout(src_embedding + self.get_pe(src_boxes))


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, emb_size, nhead, max_width, max_height,
                 num_encoder_layers, num_decoder_layers,
                 src_vocab_size, tgt_vocab_size,
                 dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.src_pe = SourcePE(emb_size, max_width, max_height, dropout=dropout)
        self.tgt_pe = TargetPE(emb_size, dropout=dropout)

    def forward(self, src, src_boxes, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.src_pe(self.src_tok_emb(src, src_boxes))
        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_INDEX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_INDEX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
