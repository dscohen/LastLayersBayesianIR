import numpy as np
from typing import Dict, List, Tuple, Optional, overload

import pdb

import torch
from torch import Tensor

from torch.nn.modules.linear import Linear

from allennlp.models import Model
from allennlp.nn import util

from transformers import BertModel
from models.layers.concete_dropout import ConcreteDropout

class UncertaintyBert (Model):
    def __init__(self,
                 bert: BertModel,num_models:int, device=-1):
        super(UncertaintyBert, self).__init__(vocab=None) # (?)
        print ('Loading Stochastic Bert')
        w,d = 1e-6,1e-3
        self.num_models = num_models
        self.device =device
        self._bert = bert
        self._embedding_size = self._bert.config.hidden_size
        self._stochastic_layer = torch.nn.Linear(self._embedding_size,self._embedding_size,bias=True)
        self._output_projection_layer = torch.nn.Linear(self._embedding_size, 2)
        self.cd1 = ConcreteDropout(weight_regulariser=w,dropout_regulariser=d)
        self.cd2 = ConcreteDropout(weight_regulariser=w,dropout_regulariser=d)
        self.m = torch.nn.ReLU()

    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor]) -> torch.Tensor:


        max_input_length = 512

        bsz = document["tokens"].size(0)
        # doc_max_length = document["tokens"].size(1)
        # qry_max_length = query["tokens"].size(1)

        tok_seq = torch.full((bsz, max_input_length), 0, dtype=int)
        seg_mask = torch.full((bsz, max_input_length), 0, dtype=int)
        if self.device !=-1:
            tok_seq = tok_seq.cuda()
            seg_mask = seg_mask.cuda()
        seg_1_value = 0
        seg_2_value = 1
        CLS_id = 101
        SEP_id = 102

        tok_seq[:, 0] = torch.full((bsz, 1), CLS_id, dtype=int)[:, 0]
        seg_mask[:, 0] = torch.full((bsz, 1), seg_1_value, dtype=int)[:, 0]

        for batch_i in range(bsz):
            # query
            _offset = 1
            _vec = query["tokens"][batch_i]
            _length = len(_vec[_vec != 0])
            tok_seq[batch_i, _offset:_length+_offset] = query["tokens"][batch_i, :_length]
            seg_mask[batch_i, _offset:_length+_offset] = torch.full((_length, 1), seg_1_value, dtype=int)[:, 0]
            _offset += _length

            tok_seq[batch_i, _offset:_offset+1] = SEP_id
            seg_mask[batch_i, _offset:_offset+1] = seg_1_value
            _offset += 1

            # document
            ## we assume that length of query (+2) never exceeds <max_input_length>
            ## therefore we only truncate the document
            ## in extreme cases this can hurt
            _vec = document["tokens"][batch_i]
            _length = len(_vec[_vec != 0])
            _fill_until = _length + _offset
            if _fill_until >= max_input_length:
                _fill_until = max_input_length - 1 # leaving space for the last <sep>
                _length = _fill_until - _offset
            tok_seq[batch_i, _offset:_fill_until] = document["tokens"][batch_i, :_length]
            seg_mask[batch_i, _offset:_fill_until] = torch.full((_length, 1), seg_2_value, dtype=int)[:, 0]
            _offset += _length

            tok_seq[batch_i, _offset:_offset+1] = SEP_id
            seg_mask[batch_i, _offset:_offset+1] = seg_2_value
            _offset += 1


        pad_mask = util.get_text_field_mask({"tokens":tok_seq})
        if self.device != -1:
            pad_mask = pad_mask.cuda()

        out = self._bert(input_ids=tok_seq, attention_mask=pad_mask, token_type_ids=seg_mask)

        #stochastic  sampling
        logprobs = []
        rels = []
        # Default to one if we're training
        if self.training:
            num_iters = 1
        else:
            num_iters = self.num_models

        for _ in range(num_iters):
            out_features = out[0][:,0,:]
            out_features = self.cd1(out_features,torch.nn.Sequential(self._stochastic_layer, self.m))
            scores = self.cd2(out_features,torch.nn.Sequential(self._output_projection_layer))
            lprobs = torch.nn.LogSoftmax(dim=-1)(scores)
            logprobs.append(lprobs)
            rels.append(lprobs[:,0].unsqueeze(1))

        if self.training:
            logprobs = logprobs[0]
            rels = rels[0]
        else:
            logprobs = torch.cat(logprobs,0)
            rels = torch.cat(rels,1)

        return {"rels" : rels.squeeze(), "logprobs": logprobs}


