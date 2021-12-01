'''RNN operators module, include RNN, GRU, LSTM'''
import math
import numpy as np
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.common import dtype as mstype
from mindspore.ops.primitive import constexpr
from mindspore import Tensor, Parameter, ParameterTuple
from mindspore import log as logger
from mindspore import context
from mindspore._checkparam import Validator as validator
from .rnn_cells import rnn_relu_cell, rnn_tanh_cell, gru_cell, lstm_cell
from .rnn_utils import Reverse, ReverseSequence

@constexpr
def _check_input_dtype(input_dtype, param_name, allow_dtypes, cls_name):
    validator.check_type_name(param_name, input_dtype, allow_dtypes, cls_name)


@constexpr
def _check_is_tensor(param_name, input_data, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and not isinstance(P.typeof(input_data), mstype.tensor_type):
        raise TypeError(f"For '{cls_name}', the '{param_name}' should be '{mstype.tensor_type}', "
                        f"but got '{P.typeof(input_data)}'")

@constexpr
def _check_is_tuple(param_name, input_data, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and not isinstance(P.typeof(input_data), mstype.Tuple):
        raise TypeError(f"For '{cls_name}', the '{param_name}' should be '{mstype.Tuple}', "
                        f"but got '{P.typeof(input_data)}'")

@constexpr
def _check_tuple_length(param_name, input_data, length, cls_name):
    """Internal function, used to check whether the input data is Tensor."""
    if input_data is not None and len(input_data) != length:
        raise TypeError(f"For '{cls_name}', the length of '{param_name}' should be '{length}', "
                        f"but got '{len(input_data)}'")

@constexpr
def _init_state(shape, dtype, is_lstm):
    hx = Tensor(np.zeros(shape), dtype)
    cx = Tensor(np.zeros(shape), dtype)
    if is_lstm:
        return (hx, cx)
    return hx

@constexpr
def arange(start, stop, step):
    return Tensor(np.arange(start, stop, step), mstype.int32)

def sequence_mask(lengths, maxlen):
    """generate mask matrix by seq_length"""
    range_vector = arange(0, maxlen, 1)
    result = range_vector < lengths.view(lengths.shape + (1,))
    return result.astype(mstype.int32)

def select_by_mask(inputs, mask):
    """mask hiddens by mask matrix"""
    return mask.view(mask.shape + (1,)).swapaxes(0, 1) \
        .expand_as(inputs).astype(mstype.bool_)  * inputs

def get_hidden(output, seq_length):
    """get hidden state by seq_length"""
    batch_index = arange(0, seq_length.shape[0], 1)
    indices = P.Concat(1)((seq_length.view(-1, 1) - 1, batch_index.view(-1, 1)))
    return P.GatherNd()(output, indices)

class _DynamicRNNBase(nn.Cell):
    '''Dynamic RNN module to compute RNN cell by timesteps'''
    def __init__(self, mode):
        super().__init__()
        if mode == "RNN_RELU":
            cell = rnn_relu_cell
        elif mode == "RNN_TANH":
            cell = rnn_tanh_cell
        elif mode == "LSTM":
            cell = lstm_cell
        elif mode == "GRU":
            cell = gru_cell
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)
        self.cell = cell
        self.is_lstm = mode == "LSTM"

    def recurrent(self, x, h_0, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps without sequence length'''
        time_step = x.shape[0]
        outputs = []
        t = 0
        h = h_0
        while t < time_step:
            x_t = x[t:t+1:1]
            x_t = P.Squeeze(0)(x_t)
            h = self.cell(x_t, h, w_ih, w_hh, b_ih, b_hh)
            if self.is_lstm:
                outputs.append(h[0])
            else:
                outputs.append(h)
            t += 1
        outputs = P.Stack()(outputs)
        return outputs, h

    def variable_recurrent(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        '''recurrent steps with sequence length'''
        time_step = x.shape[0]
        h_t = h
        if self.is_lstm:
            hidden_size = h[0].shape[-1]
            zero_output = P.ZerosLike()(h_t[0])
        else:
            hidden_size = h.shape[-1]
            zero_output = P.ZerosLike()(h_t)
        seq_length = P.Cast()(seq_length, mstype.float32)
        seq_length = P.BroadcastTo((hidden_size, -1))(seq_length)
        seq_length = P.Cast()(seq_length, mstype.int32)
        seq_length = P.Transpose()(seq_length, (1, 0))

        outputs = []
        state_t = h_t
        t = 0
        while t < time_step:
            x_t = x[t:t+1:1]
            x_t = P.Squeeze(0)(x_t)
            h_t = self.cell(x_t, state_t, w_ih, w_hh, b_ih, b_hh)
            seq_cond = seq_length > t
            if self.is_lstm:
                state_t_0 = P.Select()(seq_cond, h_t[0], state_t[0])
                state_t_1 = P.Select()(seq_cond, h_t[1], state_t[1])
                output = P.Select()(seq_cond, h_t[0], zero_output)
                state_t = (state_t_0, state_t_1)
            else:
                state_t = P.Select()(seq_cond, h_t, state_t)
                output = P.Select()(seq_cond, h_t, zero_output)
            outputs.append(output)
            t += 1
        outputs = P.Stack()(outputs)
        return outputs, state_t

    def construct(self, x, h, seq_length, w_ih, w_hh, b_ih, b_hh):
        if seq_length is None:
            return self.recurrent(x, h, w_ih, w_hh, b_ih, b_hh)
        return self.variable_recurrent(x, h, seq_length, w_ih, w_hh, b_ih, b_hh)

class _DynamicRNN_Relu(_DynamicRNNBase):
    def __init__(self):
        mode = 'RNN_RELU'
        super().__init__(mode)

class _DynamicRNN_Tanh(_DynamicRNNBase):
    def __init__(self):
        mode = 'RNN_TANH'
        super().__init__(mode)

class _DynamicGRU_CPU_GPU(_DynamicRNNBase):
    def __init__(self):
        mode = 'GRU'
        super().__init__(mode)

class _DynamicGRU_Ascend(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gru = P.DynamicGRUV2(gate_order='rzh')
        self.transpose = P.Transpose()
        self.dtype = mstype.float16

    def construct(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        if b_ih is None:
            b_ih = P.Zeros()(w_ih.shape[0], w_ih.dtype)
            b_hh = P.Zeros()(w_ih.shape[0], w_ih.dtype)
        outputs, _, _, _, _, _ = self.gru(self.cast(x, self.dtype), \
                                         self.cast(self.transpose(w_ih, (1, 0)), self.dtype), \
                                         self.cast(self.transpose(w_hh, (1, 0)), self.dtype), \
                                         self.cast(b_ih, self.dtype), \
                                         self.cast(b_hh, self.dtype), \
                                         None, self.cast(h_0, self.dtype))
        if seq_length is not None:
            h = get_hidden(outputs, seq_length)
            mask = sequence_mask(seq_length, x.shape[0])
            outputs = select_by_mask(outputs, mask)
        else:
            h = outputs[-1]
        return outputs, h

class _DynamicLSTM_CPU_GPU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.concat = P.Concat()
        self.is_gpu = context.get_context("device_target") == "GPU"

    def construct(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        gate_size, input_size = w_ih.shape
        hidden_size = gate_size // 4
        if self.is_gpu and seq_length is None:
            if b_ih is None:
                weights = self.concat((
                    w_ih.view(-1, 1, 1),
                    w_hh.view(-1, 1, 1)
                ))
                has_bias = False
            else:
                weights = self.concat((
                    w_ih.view(-1, 1, 1),
                    w_hh.view(-1, 1, 1),
                    b_ih.view(-1, 1, 1),
                    b_hh.view(-1, 1, 1)
                ))
                has_bias = True
            output, h_n, c_n, _, _ = P.LSTM(input_size, hidden_size, 1, has_bias, False, 0.0)(
                x,
                h_0[0].view(1, *h_0[0].shape),
                h_0[1].view(1, *h_0[1].shape),
                weights
            )
        else:
            output, (h_n, c_n) = _DynamicRNNBase('LSTM')(x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh)
        return output, (h_n, c_n)
class _DynamicLSTM_Ascend(nn.Cell):
    def __init__(self):
        super().__init__()
        self.lstm = P.DynamicRNN()
        self.concat_dim1 = P.Concat(axis=1)
        self.concat_dim0 = P.Concat(axis=0)
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.split = P.Split(axis=0, output_num=4)
        self.dtype = mstype.float16

    def construct(self, x, h_0, seq_length, w_ih, w_hh, b_ih, b_hh):
        w_ih_i, w_ih_f, w_ih_g, w_ih_o = self.split(w_ih)
        w_hh_i, w_hh_f, w_hh_g, w_hh_o = self.split(w_hh)
        w_ih = self.concat_dim0((w_ih_i, w_ih_g, w_ih_f, w_ih_o))
        w_hh = self.concat_dim0((w_hh_i, w_hh_g, w_hh_f, w_hh_o))
        weight = self.concat_dim1((w_ih, w_hh))
        if b_ih is None:
            bias = P.Zeros()(w_ih.shape[0], w_ih.dtype)
        else:
            b_ih_i, b_ih_f, b_ih_g, b_ih_o = self.split(b_ih)
            b_hh_i, b_hh_f, b_hh_g, b_hh_o = self.split(b_hh)
            bias = self.concat_dim0((b_ih_i + b_hh_i, \
                                     b_ih_g + b_hh_g, \
                                     b_ih_f + b_hh_f, \
                                     b_ih_o + b_hh_o))

        outputs, h, c, _, _, _, _, _ = self.lstm(self.cast(x, self.dtype), \
                                                 self.cast(self.transpose(weight, (1, 0)), self.dtype), \
                                                 self.cast(bias, self.dtype), None, \
                                                 self.cast(h_0[0].view(1, *h_0[0].shape), self.dtype), \
                                                 self.cast(h_0[1].view(1, *h_0[1].shape), self.dtype))
        if seq_length is not None:
            h = get_hidden(h, seq_length)
            c = get_hidden(c, seq_length)
            mask = sequence_mask(seq_length, x.shape[0])
            outputs = select_by_mask(outputs, mask)
        else:
            h = h[-1]
            c = c[-1]
        return outputs, (h, c)

class _RNNBase(nn.Cell):
    '''Basic class for RNN operators'''
    def __init__(self, mode, input_size, hidden_size, num_layers=1, has_bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        is_ascend = context.get_context("device_target") == "Ascend"
        if not 0 <= dropout <= 1:
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")

        if dropout > 0 and num_layers == 1:
            logger.warning("dropout option adds dropout after all but last "
                           "recurrent layer, so non-zero dropout expects "
                           "num_layers greater than 1, but got dropout={} and "
                           "num_layers={}".format(dropout, num_layers))
        if mode == "LSTM":
            gate_size = 4 * hidden_size
            self.rnn = _DynamicLSTM_Ascend() if is_ascend else _DynamicLSTM_CPU_GPU()
        elif mode == "GRU":
            gate_size = 3 * hidden_size
            self.rnn = _DynamicGRU_Ascend() if is_ascend else _DynamicGRU_CPU_GPU()
        elif mode == "RNN_TANH":
            gate_size = hidden_size
            self.rnn = _DynamicRNN_Tanh()
        elif mode == "RNN_RELU":
            gate_size = hidden_size
            self.rnn = _DynamicRNN_Relu()
        else:
            raise ValueError("Unrecognized RNN mode: " + mode)

        if context.get_context("device_target") == "CPU":
            self.reverse = Reverse(0)
            self.reverse_sequence = ReverseSequence(0, 1)
        else:
            self.reverse = P.ReverseV2([0])
            self.reverse_sequence = P.ReverseSequence(0, 1)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_op = nn.Dropout(float(1 - dropout))
        self.bidirectional = bidirectional
        self.has_bias = has_bias
        num_directions = 2 if bidirectional else 1
        self.is_lstm = mode == "LSTM"

        self.w_ih_list = []
        self.w_hh_list = []
        self.b_ih_list = []
        self.b_hh_list = []
        stdv = 1 / math.sqrt(self.hidden_size)
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                self.w_ih_list.append(Parameter(
                    Tensor(np.random.uniform(-stdv, stdv, (gate_size, layer_input_size)).astype(np.float32)),
                    name='weight_ih_l{}{}'.format(layer, suffix)))
                self.w_hh_list.append(Parameter(
                    Tensor(np.random.uniform(-stdv, stdv, (gate_size, hidden_size)).astype(np.float32)),
                    name='weight_hh_l{}{}'.format(layer, suffix)))
                if has_bias:
                    self.b_ih_list.append(Parameter(
                        Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_ih_l{}{}'.format(layer, suffix)))
                    self.b_hh_list.append(Parameter(
                        Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32)),
                        name='bias_hh_l{}{}'.format(layer, suffix)))
        self.w_ih_list = ParameterTuple(self.w_ih_list)
        self.w_hh_list = ParameterTuple(self.w_hh_list)
        self.b_ih_list = ParameterTuple(self.b_ih_list)
        self.b_hh_list = ParameterTuple(self.b_hh_list)

    def _stacked_bi_dynamic_rnn(self, x, h, seq_length):
        """stacked bidirectional dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            offset = i * 2
            if self.has_bias:
                w_f_ih, w_f_hh, b_f_ih, b_f_hh = \
                    self.w_ih_list[offset], self.w_hh_list[offset], \
                    self.b_ih_list[offset], self.b_hh_list[offset]
                w_b_ih, w_b_hh, b_b_ih, b_b_hh = \
                    self.w_ih_list[offset + 1], self.w_hh_list[offset + 1], \
                    self.b_ih_list[offset + 1], self.b_hh_list[offset + 1]
            else:
                w_f_ih, w_f_hh = self.w_ih_list[offset], self.w_hh_list[offset]
                w_b_ih, w_b_hh = self.w_ih_list[offset + 1], self.w_hh_list[offset + 1]
                b_f_ih, b_f_hh, b_b_ih, b_b_hh = None, None, None, None
            if self.is_lstm:
                h_f_i = (h[0][offset], h[1][offset])
                h_b_i = (h[0][offset + 1], h[1][offset + 1])
            else:
                h_f_i = h[offset]
                h_b_i = h[offset + 1]
            if seq_length is None:
                x_b = self.reverse(pre_layer)
            else:
                x_b = self.reverse_sequence(pre_layer, seq_length)
            output_f, h_t_f = self.rnn(pre_layer, h_f_i, seq_length, w_f_ih, w_f_hh, b_f_ih, b_f_hh)
            output_b, h_t_b = self.rnn(x_b, h_b_i, seq_length, w_b_ih, w_b_hh, b_b_ih, b_b_hh)
            if seq_length is None:
                output_b = self.reverse(output_b)
            else:
                output_b = self.reverse_sequence(output_b, seq_length)
            output = P.Concat(2)((output_f, output_b))
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t_f[0], h_t_b[0],)
                c_n += (h_t_f[1], h_t_b[1],)
            else:
                h_n += (h_t_f, h_t_b,)
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))
        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def _stacked_dynamic_rnn(self, x, h, seq_length):
        """stacked mutil_layer dynamic_rnn"""
        pre_layer = x
        h_n = ()
        c_n = ()
        output = 0
        for i in range(self.num_layers):
            if self.has_bias:
                w_ih, w_hh, b_ih, b_hh = self.w_ih_list[i], self.w_hh_list[i], self.b_ih_list[i], self.b_hh_list[i]
            else:
                w_ih, w_hh = self.w_ih_list[i], self.w_hh_list[i]
                b_ih, b_hh = None, None
            if self.is_lstm:
                h_i = (h[0][i], h[1][i])
            else:
                h_i = h[i]
            output, h_t = self.rnn(pre_layer, h_i, seq_length, w_ih, w_hh, b_ih, b_hh)
            pre_layer = self.dropout_op(output) if (self.dropout != 0 and i < self.num_layers - 1) else output
            if self.is_lstm:
                h_n += (h_t[0],)
                c_n += (h_t[1],)
            else:
                h_n += (h_t,)
        if self.is_lstm:
            h_n = P.Concat(0)(h_n)
            c_n = P.Concat(0)(c_n)
            h_n = h_n.view(h[0].shape)
            c_n = c_n.view(h[1].shape)
            return output, (h_n.view(h[0].shape), c_n.view(h[1].shape))
        h_n = P.Concat(0)(h_n)
        return output, h_n.view(h.shape)

    def construct(self, x, hx=None, seq_length=None):
        '''Defines the RNN like operators performed'''
        _check_is_tensor("x", x, self.cls_name)
        _check_input_dtype(x.dtype, "x", [mstype.float32], self.cls_name)
        if hx is not None:
            if not self.is_lstm:
                _check_is_tensor("hx", hx, self.cls_name)
                _check_input_dtype(hx.dtype, "hx", [mstype.float32], self.cls_name)
            else:
                _check_is_tuple('hx', hx, self.cls_name)
                _check_tuple_length('hx', hx, 2, self.cls_name)
                _check_is_tensor('hx[0]', hx[0], self.cls_name)
                _check_is_tensor('hx[1]', hx[1], self.cls_name)
                _check_input_dtype(hx[0].dtype, "hx[0]", [mstype.float32], self.cls_name)
                _check_input_dtype(hx[1].dtype, "hx[1]", [mstype.float32], self.cls_name)
        if seq_length is not None:
            _check_input_dtype(seq_length.dtype, "seq_length", [mstype.int32, mstype.int64], self.cls_name)
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        num_directions = 2 if self.bidirectional else 1
        if hx is None:
            hx = _init_state((self.num_layers * num_directions, max_batch_size, self.hidden_size), x.dtype, self.is_lstm)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        if self.bidirectional:
            x, h = self._stacked_bi_dynamic_rnn(x, hx, seq_length)
        else:
            x, h = self._stacked_dynamic_rnn(x, hx, seq_length)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
        return x, h

class RNN(_RNNBase):
    r"""
    Stacked Elman RNN layers.

    Apply RNN layer with :math:`\tanh` or :math:`\text{ReLU}` non-linearity to the input.

    For each element in the input sequence, each layer computes the following function:

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    Here :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``'relu'``, then :math:`\text{ReLU}` is used instead of :math:`\tanh`.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked RNN. Default: 1.
        nonlinearity (str): The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            RNN layer except the last layer. Default 0.0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional RNN,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and
          shape (num_directions * `num_layers`, batch_size, `hidden_size`). Data type of `hx` must be the same as `x`.
        - **seq_length** (Tensor) - The length of each sequence in a input batch.
          Tensor of shape :math:`(\text{batch_size})`. Default: None.
          This input indicates the real sequence length before padding to avoid padded elements
          have been used to compute hidden state and affect the final output. It is recommend to
          use this input when **x** has padding elements.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`) or
          (batch_size, seq_len, num_directions * `hidden_size`).
        - **hx_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0).
        ValueError: If `nonlinearity` is not in ['tanh', 'relu'].

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = nn.RNN(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """
    def __init__(self, *args, **kwargs):
        if 'nonlinearity' in kwargs:
            if kwargs['nonlinearity'] == 'tanh':
                mode = 'RNN_TANH'
            elif kwargs['nonlinearity'] == 'relu':
                mode = 'RNN_RELU'
            else:
                raise ValueError("Unknown nonlinearity '{}'".format(
                    kwargs['nonlinearity']))
            del kwargs['nonlinearity']
        else:
            mode = 'RNN_TANH'

        super(RNN, self).__init__(mode, *args, **kwargs)

class GRU(_RNNBase):
    r"""
    Stacked GRU (Gated Recurrent Unit) layers.

    Apply GRU layer to the input.

    There are two gates in a GRU model; one is update gate and the other is reset gate.
    Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}`, the update and reset gate at
    time :math:`t` is computed using an gating mechanism. Update gate :math:`z_t` is designed to protect the cell
    from perturbation by irrelevant inputs and past hidden state. Reset gate :math:`r_t` determines how much
    information should be reset from old hidden state. New memory state :math:`{n}_t` is
    calculated with the current input, on which the reset gate will be applied. Finally, current hidden state
    :math:`h_{t}` is computed with the calculated update grate and new memory state. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ir}, b_{ir}` are the weight and bias used to transform from input :math:`x` to :math:`r`.
    Details can be found in paper
    `Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
    <https://aclanthology.org/D14-1179.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked GRU. Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float): If not 0.0, append `Dropout` layer on the outputs of each
            GRU layer except the last layer. Default 0.0. The range of dropout is [0.0, 1.0).
        bidirectional (bool): Specifies whether it is a bidirectional GRU,
            num_directions=2 if bidirectional=True otherwise 1. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of data type mindspore.float32 and
          shape (seq_len, batch_size, `input_size`) or (batch_size, seq_len, `input_size`).
        - **hx** (Tensor) - Tensor of data type mindspore.float32 and
          shape (num_directions * `num_layers`, batch_size, `hidden_size`). Data type of `hx` must be the same as `x`.
        - **seq_length** (Tensor) - The length of each sequence in a input batch.
          Tensor of shape :math:`(\text{batch_size})`. Default: None.
          This input indicates the real sequence length before padding to avoid padded elements
          have been used to compute hidden state and affect the final output. It is recommend to
          use this input when **x** has padding elements.

    Outputs:
        Tuple, a tuple contains (`output`, `h_n`).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`) or
          (batch_size, seq_len, num_directions * `hidden_size`).
        - **hx_n** (Tensor) - Tensor of shape (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.GRU(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, hn = net(x, h0)
        >>> print(output.shape)
        (3, 5, 16)
    """
    def __init__(self, *args, **kwargs):
        mode = 'GRU'
        super(GRU, self).__init__(mode, *args, **kwargs)

class LSTM(_RNNBase):
    r"""
    Stacked LSTM (Long Short-Term Memory) layers.

    Apply LSTM layer to the input.

    There are two pipelines connecting two consecutive cells in a LSTM model; one is cell state pipeline
    and the other is hidden state pipeline. Denote two consecutive time nodes as :math:`t-1` and :math:`t`.
    Given an input :math:`x_t` at time :math:`t`, an hidden state :math:`h_{t-1}` and an cell
    state :math:`c_{t-1}` of the layer at time :math:`{t-1}`, the cell state and hidden state at
    time :math:`t` is computed using an gating mechanism. Input gate :math:`i_t` is designed to protect the cell
    from perturbation by irrelevant inputs. Forget gate :math:`f_t` affords protection of the cell by forgetting
    some information in the past, which is stored in :math:`h_{t-1}`. Output gate :math:`o_t` protects other
    units from perturbation by currently irrelevant memory contents. Candidate cell state :math:`\tilde{c}_t` is
    calculated with the current input, on which the input gate will be applied. Finally, current cell state
    :math:`c_{t}` and hidden state :math:`h_{t}` are computed with the calculated gates and cell states. The complete
    formulation is as follows.

    .. math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ix} x_t + b_{ix} + W_{ih} h_{(t-1)} + b_{ih}) \\
            f_t = \sigma(W_{fx} x_t + b_{fx} + W_{fh} h_{(t-1)} + b_{fh}) \\
            \tilde{c}_t = \tanh(W_{cx} x_t + b_{cx} + W_{ch} h_{(t-1)} + b_{ch}) \\
            o_t = \sigma(W_{ox} x_t + b_{ox} + W_{oh} h_{(t-1)} + b_{oh}) \\
            c_t = f_t * c_{(t-1)} + i_t * \tilde{c}_t \\
            h_t = o_t * \tanh(c_t) \\
        \end{array}

    Here :math:`\sigma` is the sigmoid function, and :math:`*` is the Hadamard product. :math:`W, b`
    are learnable weights between the output and the input in the formula. For instance,
    :math:`W_{ix}, b_{ix}` are the weight and bias used to transform from input :math:`x` to :math:`i`.
    Details can be found in paper `LONG SHORT-TERM MEMORY
    <https://www.bioinf.jku.at/publications/older/2604.pdf>`_ and
    `Long Short-Term Memory Recurrent Neural Network Architectures for Large Scale Acoustic Modeling
    <https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf>`_.

    Args:
        input_size (int): Number of features of input.
        hidden_size (int):  Number of features of hidden layer.
        num_layers (int): Number of layers of stacked LSTM . Default: 1.
        has_bias (bool): Whether the cell has bias `b_ih` and `b_hh`. Default: True.
        batch_first (bool): Specifies whether the first dimension of input `x` is batch_size. Default: False.
        dropout (float, int): If not 0, append `Dropout` layer on the outputs of each
            LSTM layer except the last layer. Default 0. The range of dropout is [0.0, 1.0].
        bidirectional (bool): Specifies whether it is a bidirectional LSTM. Default: False.

    Inputs:
        - **x** (Tensor) - Tensor of shape (seq_len, batch_size, `input_size`) or
          (batch_size, seq_len, `input_size`).
        - **hx** (tuple) - A tuple of two Tensors (h_0, c_0) both of data type mindspore.float32 or
          mindspore.float16 and shape (num_directions * `num_layers`, batch_size, `hidden_size`).
          Data type of `hx` must be the same as `x`.
        - **seq_length** (Tensor) - The length of each sequence in a input batch.
          Tensor of shape :math:`(\text{batch_size})`. Default: None.
          This input indicates the real sequence length before padding to avoid padded elements
          have been used to compute hidden state and affect the final output. It is recommend to
          use this input when **x** has padding elements.

    Outputs:
        Tuple, a tuple contains (`output`, (`h_n`, `c_n`)).

        - **output** (Tensor) - Tensor of shape (seq_len, batch_size, num_directions * `hidden_size`).
        - **hx_n** (tuple) - A tuple of two Tensor (h_n, c_n) both of shape
          (num_directions * `num_layers`, batch_size, `hidden_size`).

    Raises:
        TypeError: If `input_size`, `hidden_size` or `num_layers` is not an int.
        TypeError: If `has_bias`, `batch_first` or `bidirectional` is not a bool.
        TypeError: If `dropout` is neither a float nor an int.
        ValueError: If `dropout` is not in range [0.0, 1.0].

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> net = nn.LSTM(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
        >>> x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
        >>> h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> c0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
        >>> output, (hn, cn) = net(x, (h0, c0))
        >>> print(output.shape)
        (3, 5, 16)
    """
    def __init__(self, *args, **kwargs):
        mode = 'LSTM'
        super(LSTM, self).__init__(mode, *args, **kwargs)