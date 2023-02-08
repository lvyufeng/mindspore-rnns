import unittest
import math
import numpy as np
from ddt import ddt, data
import mindspore
from mindspore import Tensor
from rnns.rnn_cells_fused import MultiLayerLSTM

@ddt
class TestMultiLayerLSTM(unittest.TestCase):
    @data(False, True)
    def test_forward_one_layer_seq_length_is_none(self, jit):
        input_size = 100
        hidden_size = 200
        num_layers = 1
        seq_len = 500
        batch_size = 32
        gate_size = 4
        stdv = 1 / math.sqrt(hidden_size)
        w_ih = Tensor(np.random.uniform(-stdv, stdv, (gate_size, input_size)).astype(np.float32))
        w_hh = Tensor(np.random.uniform(-stdv, stdv, (gate_size, hidden_size)).astype(np.float32))
        b_ih = Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32))
        b_hh = Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32))

        net = MultiLayerLSTM(input_size, hidden_size, num_layers, True, 0.0)
        input_tensor = Tensor(np.ones([seq_len, batch_size, input_size]).astype(np.float32))
        h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        c0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        def forward(input_tensor, h0, c0, w_ih, w_hh, b_ih, b_hh):
            output, hn, cn = net(input_tensor, h0, c0, None, w_ih, w_hh, b_ih, b_hh)
            return output, hn, cn
        
        if jit:
            forward = mindspore.jit(forward)

        output, hn, cn = forward(input_tensor, h0, c0, w_ih, w_hh, b_ih, b_hh)
        print(input_tensor.shape, output.shape)
        print(h0.shape, hn.shape)
        print(c0.shape, cn.shape)
