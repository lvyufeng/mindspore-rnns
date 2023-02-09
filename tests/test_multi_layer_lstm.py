import unittest
import math
import numpy as np
from ddt import ddt, data, unpack
import mindspore
from mindspore import Tensor, Parameter
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
        def forward(input_tensor, h0, c0, weight_list):
            output, hn, cn = net(input_tensor, h0, c0, None, weight_list)
            return output, hn, cn
        
        if jit:
            forward = mindspore.jit(forward)

        output, hn, cn = forward(input_tensor, h0, c0, (w_ih, w_hh, b_ih, b_hh))
        print(input_tensor.shape, output.shape)
        print(h0.shape, hn.shape)
        print(c0.shape, cn.shape)


    @unpack
    @data(
        {'num_layers': 2, 'jit': True},
        {'num_layers': 2, 'jit': False},
    )
    def test_forward_multi_layer_seq_length_is_none(self, jit, num_layers):
        input_size = 10
        hidden_size = 20
        seq_len = 5
        batch_size = 2
        gate_size = 4
        num_directions = 1
        stdv = 1 / math.sqrt(hidden_size)
        all_weights = ()

        weight_size = 0
        for layer in range(num_layers):
            input_layer_size = input_size if layer == 0 else hidden_size * num_directions
            increment_size = gate_size * input_layer_size
            increment_size += gate_size * hidden_size
            # increment_size += 1 * gate_size
            weight_size += increment_size * num_directions
        
        print(weight_size)
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * num_directions
            w_ih = Tensor(np.random.uniform(-stdv, stdv, (gate_size, layer_input_size)).astype(np.float32))
            w_hh = Tensor(np.random.uniform(-stdv, stdv, (gate_size, hidden_size)).astype(np.float32))
            # b_ih = Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32))
            # b_hh = Tensor(np.random.uniform(-stdv, stdv, (gate_size)).astype(np.float32))
            all_weights += (w_ih, w_hh)#, b_ih, b_hh)

        net = MultiLayerLSTM(input_size, hidden_size, num_layers, False, 0.0)
        input_tensor = Tensor(np.ones([seq_len, batch_size, input_size]).astype(np.float32))
        h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        c0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))

        w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)
        w = Parameter(Tensor(w_np), 'w')
        def forward(input_tensor, h0, c0, all_weights):
            output, hn, cn = net(input_tensor, h0, c0, None, all_weights)
            return output, hn, cn
        
        if jit:
            forward = mindspore.jit(forward)

        output, hn, cn = forward(input_tensor, h0, c0, w)
        print(input_tensor.shape, output.shape)
        print(h0.shape, hn.shape)
        print(c0.shape, cn.shape)

    def test_op(self):
        from mindspore import ops
        input_size = 10
        hidden_size = 2
        num_layers = 1
        seq_len = 5
        batch_size = 2

        net = ops.LSTM(input_size, hidden_size, num_layers, True, False, 0.0)
        input_tensor = Tensor(np.ones([seq_len, batch_size, input_size]).astype(np.float32))
        h0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        c0 = Tensor(np.ones([num_layers, batch_size, hidden_size]).astype(np.float32))
        w = Tensor(np.ones([1000, 1, 1]).astype(np.float32))
        output, hn, cn, _, _ = net(input_tensor, h0, c0, w)
        print(output)










