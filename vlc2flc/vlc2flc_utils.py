import math
import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence

from utils import get_flc2tok, SOS_INDEX, EOS_INDEX, PAD_INDEX
from vlc2flc.vlc2flc_configs import MODEL_PATH


class TargetPE(nn.Module):
    def __init__(self, emb_size: int, dropout: float, device, max_len: int = 500):
        super(TargetPE, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size, 2).to(device) * math.log(10000) / emb_size)
        pos = torch.arange(0, max_len).to(device).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, emb_size)).to(device)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, tgt_embedding):
        return self.dropout(tgt_embedding + self.pos_embedding[:tgt_embedding.size(0), :])


class SourcePE(nn.Module):
    def __init__(self, emb_size, max_width, max_height, dropout, device):
        self.device = device
        super(SourcePE, self).__init__()

        den = torch.exp(-torch.arange(0, emb_size / 4).to(device) * math.log(10000) / emb_size / 4)

        x_p = torch.arange(0, max_width + 1).to(device).reshape(max_width + 1, 1)
        x_pe = torch.sin(x_p * den)

        y_p = torch.arange(0, max_height + 1).to(device).reshape(max_height + 1, 1)
        y_pe = torch.sin(y_p * den)

        self.register_buffer('x_pe', x_pe)
        self.register_buffer('y_pe', y_pe)

        self.emb_size = emb_size

        self.dropout = nn.Dropout(dropout)

    def get_pe(self, src_boxes_list):
        pe = torch.zeros((len(src_boxes_list[0]), len(src_boxes_list), self.emb_size),
                         requires_grad=False).to(self.device)
        for index, src_boxes in enumerate(src_boxes_list):
            for box_index, box in enumerate(src_boxes):
                pe[box_index, index, 0::4] = self.x_pe[int(box[0])]
                pe[box_index, index, 1::4] = self.y_pe[int(box[1])]
                pe[box_index, index, 2::4] = self.x_pe[int(box[2])]
                pe[box_index, index, 3::4] = self.y_pe[int(box[3])]

        return pe

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
                 src_vocab_size, tgt_vocab_size, device,
                 dim_feedforward=512, dropout=0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       device=device)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.src_pe = SourcePE(emb_size, max_width, max_height, dropout=dropout, device=device)
        self.tgt_pe = TargetPE(emb_size, dropout=dropout, device=device)
        self.device = device

    def forward(self, src, src_boxes, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask):
        tgt_emb = self.tgt_pe(self.tgt_tok_emb(tgt))
        src_emb = self.src_pe(self.src_tok_emb(src), src_boxes)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_boxes, src_mask: Tensor):
        return self.transformer.encoder(self.src_pe(self.src_tok_emb(src), src_boxes), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.tgt_pe(self.tgt_tok_emb(tgt)), memory, tgt_mask)

    def translate(self, src, boxes):
        src_mask = torch.zeros(len(src), len(src)).type(torch.bool).to(self.device)

        memory = self.encode(src, boxes, src_mask)
        result = torch.ones(1, 1).fill_(SOS_INDEX).type(torch.long).to(self.device)
        while True:
            result_mask = (generate_square_subsequent_mask(result.size(0), self.device).type(torch.bool))
            out = self.decode(result, memory, result_mask).transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)

            result = torch.cat([result,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word.item())], dim=0)

            if next_word == EOS_INDEX:
                break

        return result


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz)).to(device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).to(device).type(torch.bool)

    src_padding_mask = (src == PAD_INDEX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_INDEX).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def tensor_transform(tokens, device):
    return torch.cat((torch.tensor([SOS_INDEX]).to(device),
                      torch.tensor(tokens).to(device),
                      torch.tensor([EOS_INDEX]).to(device)))


def get_padding_boxes(boxes_list, max_len):
    for index, boxes in enumerate(boxes_list):
        boxes_list[index] = torch.cat((torch.zeros((1, 4)),
                                       boxes,
                                       torch.tensor([[10000, 10000, 10000, 10000]]),
                                       torch.zeros((max_len - len(boxes), 4))))

    return boxes_list


def collate_fn(batch):
    src_list, tgt_list, boxes_list = [], [], []
    max_len = 0
    for src, tgt, boxes in batch:
        src_list.append(src)
        tgt_list.append(tgt)
        boxes_list.append(boxes)
        max_len = max(max_len, len(boxes))

    src_list = pad_sequence(src_list, padding_value=PAD_INDEX)
    tgt_list = pad_sequence(tgt_list, padding_value=PAD_INDEX)
    boxes_list = get_padding_boxes(boxes_list, max_len)

    return src_list, tgt_list, boxes_list


def load_checkpoint(device):
    if os.path.exists(os.path.join(MODEL_PATH,
                                   "checkpoint.pth")):
        checkpoint = torch.load(os.path.join(MODEL_PATH,
                                             "checkpoint.pth"),
                                map_location=device)
    else:
        checkpoint = None

    return checkpoint


def save_checkpoint(epoch, total_epoch, batch, total_batch, model, optimizer, scheduler):
    if not os.path.exists(os.path.join(MODEL_PATH)):
        os.makedirs(os.path.join(MODEL_PATH))

    values = {'start_epoch': epoch,
              'start_batch': batch,
              'model': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict()}

    torch.save(values, os.path.join(MODEL_PATH,
                                    "{}-{}_{}-{}.pth".format(epoch, total_epoch,
                                                             batch, total_batch)))
    torch.save(values, os.path.join(MODEL_PATH,
                                    "checkpoint.pth".format(epoch, total_epoch,
                                                            batch, total_batch)))


def get_model(device, log_model=True):
    model = Seq2SeqTransformer(128, 8, 10000, 10000, 6, 6,
                               len(get_flc2tok()) + 1, len(get_flc2tok()) + 1, device=device)

    model.to(device)

    if log_model:
        print(model)

    return model


def load_model(model, checkpoint):
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])

    return model


def load_optimizer(optimizer, checkpoint):
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer


def load_starts(checkpoint):
    if checkpoint is not None:
        start_epoch = checkpoint['start_epoch']
        start_batch = checkpoint['start_batch']
    else:
        start_epoch = 0
        start_batch = 0

    return start_epoch, start_batch


def load_scheduler(scheduler, checkpoint):
    if checkpoint is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return scheduler
