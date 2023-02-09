import math
import numpy as np
import mindspore
from mindspore import nn, ops, context
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Uniform
from mindspore.ops._primitive_cache import _get_cache_prim

def sequence_mask(lengths, maxlen):
    """generate mask matrix by seq_length"""
    range_vector = ops.arange(start=0, end=maxlen, step=1, dtype=lengths.dtype)
    result = range_vector < lengths.view(lengths.shape + (1,))
    return result.astype(mindspore.int32)

def select_by_mask(inputs, mask):
    """mask hiddens by mask matrix"""
    return mask.view(mask.shape + (1,)).swapaxes(0, 1) \
               .expand_as(inputs).astype(mindspore.bool_) * inputs

def get_hidden(output, seq_length):
    """get hidden state by seq_length"""
    batch_index = ops.arange(start=0, end=seq_length.shape[0], step=1, dtype=seq_length.dtype)
    indices = ops.concat((seq_length.view(-1, 1) - 1, batch_index.view(-1, 1)), 1)
    return ops.gather_nd(output, indices)

class MultiLSTMCell(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers=1, has_bias=False, dropout=0.):
        super().__init__()
        self.cell = ops.LSTM(input_size, hidden_size, num_layers, has_bias, False, dropout)

    def construct(self, input, hidden, cell, weight):
        # inputs: (1, batch_size, input_size)
        # h_0: (num_directions * num_layers, batch_size, hidden_size)
        # c_0: (num_directions * num_layers, batch_size, hidden_size)
        output, hidden_next, cell_next, _, _ = self.cell(input, hidden, cell, weight)
        return output, hidden_next, cell_next

class MultiLayerLSTM(nn.Cell):
    def __init__(self, input_size, hidden_size, num_layers=1, has_bias=False, dropout=0.):
        super().__init__()
        self.is_gpu = context.get_context("device_target") == "GPU"
        self.is_ascend = context.get_context('device_target') == 'Ascend'
        self.hidden_size = hidden_size
        if self.is_ascend:
            self.cell = ops.DynamicRNN(cell_depth=num_layers, keep_prob=1-dropout)
        else:
            self.cell = MultiLSTMCell(input_size, hidden_size, num_layers, has_bias, dropout)

    def construct(self, x, h_0, c_0, seq_length, weight_list):
        if self.is_ascend:
            return self._construct_ascend(x, h_0, c_0, seq_length, weight_list)
        else:
            return self._construct_gpu_cpu(x, h_0, c_0, seq_length, weight_list)

    def _construct_gpu_cpu(self, x, h_0, c_0, seq_length, weight_list):
        # inputs: (seq_length, batch_size, input_size)
        # h_0: (num_directions * num_layers, batch_size, hidden_size)
        # c_0: (num_directions * num_layers, batch_size, hidden_size)
        weights = Tensor._flatten_tensors(weight_list, 0)[0]
        max_seq_length = x.shape[0]
        output_tensor = ops.zeros(x.shape[:-1] + (self.hidden_size,), x.dtype)
        # input_list = ops.tensor_split(x, max_seq_length)
        t = Tensor(0, mindspore.int32)
        h = h_0
        c = c_0
        while t < max_seq_length:
            output, h_t, c_t = self.cell(x[t:t + 1:1], h, c, weights)
            if seq_length is not None:
                h = ops.select(seq_length, h_t, h)
                c = ops.select(seq_length, c_t, c)
            else:
                h = h_t
                c = c_t
            output_tensor[t] = output
            t += 1
        # outputs = ops.concat(outputs)
        if seq_length is not None:
            mask = sequence_mask(seq_length, x.shape[0])
            output_tensor = select_by_mask(output_tensor, mask)
        return output_tensor, h, c

    def _construct_ascend(self, x, h_0, c_0, seq_length, weight_list):
        w_ih, w_hh, b_ih, b_hh = weight_list
        w_ih_i, w_ih_f, w_ih_g, w_ih_o = ops.split(w_ih, 4)
        w_hh_i, w_hh_f, w_hh_g, w_hh_o = ops.split(w_hh, 4)
        w_ih = ops.concat((w_ih_i, w_ih_g, w_ih_f, w_ih_o))
        w_hh = ops.concat((w_hh_i, w_hh_g, w_hh_f, w_hh_o))
        weight = ops.concat((w_ih, w_hh), 1)
        if b_ih is None:
            bias = ops.zeros(w_ih.shape[0], w_ih.dtype)
        else:
            b_ih_i, b_ih_f, b_ih_g, b_ih_o = ops.split(b_ih, 4)
            b_hh_i, b_hh_f, b_hh_g, b_hh_o = ops.split(b_hh, 4)
            bias = ops.concat((b_ih_i + b_hh_i, \
                               b_ih_g + b_hh_g, \
                               b_ih_f + b_hh_f, \
                               b_ih_o + b_hh_o))

        outputs, h, c, _, _, _, _, _ = self.cell(x, \
                                                 weight.transpose(1, 0).astype(x.dtype), \
                                                 bias.astype(x.dtype), None, \
                                                 h_0.astype(x.dtype), \
                                                 c_0.astype(x.dtype))
        if seq_length is not None:
            h = get_hidden(h, seq_length)
            c = get_hidden(c, seq_length)
            mask = sequence_mask(seq_length, x.shape[0])
            outputs = select_by_mask(outputs, mask)
        else:
            h = h[-1]
            c = c[-1]
        return outputs, h, c