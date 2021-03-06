import unittest
import mindspore
import numpy as np
from rnns.rnn_cells import RNNCell
from mindspore import Tensor
import torch

class TestRNNCell(unittest.TestCase):
    def setUp(self):
        self.input_size, self.hidden_size = 16, 32
        self.x = np.random.randn(1, self.input_size)
        self.hx = np.random.randn(1, self.hidden_size)

    def test_rnn_cell(self):
        net = RNNCell(self.input_size, self.hidden_size)
        x = Tensor(self.x, mindspore.float32)
        hx = Tensor(self.hx, mindspore.float32)
        hn = net(x, hx)
        assert hn.shape == (1, self.hidden_size)
    
    def test_rnn_cell_forward(self):
        net_ms = RNNCell(self.input_size, self.hidden_size)
        x_ms = Tensor(self.x, mindspore.float32)
        hx_ms = Tensor(self.hx, mindspore.float32)
        hn_ms = net_ms(x_ms, hx_ms)

        net_pt = torch.nn.RNNCell(self.input_size, self.hidden_size)
        net_pt.weight_ih = torch.nn.Parameter(torch.Tensor(net_ms.weight_ih.asnumpy()))
        net_pt.weight_hh = torch.nn.Parameter(torch.Tensor(net_ms.weight_hh.asnumpy()))
        net_pt.bias_ih = torch.nn.Parameter(torch.Tensor(net_ms.bias_ih.asnumpy()))
        net_pt.bias_hh = torch.nn.Parameter(torch.Tensor(net_ms.bias_hh.asnumpy()))

        x_pt = torch.tensor(self.x, dtype=torch.float32)
        hx_pt = torch.tensor(self.hx, dtype=torch.float32)
        hn_pt = net_pt(x_pt, hx_pt)

        print(hn_ms.shape, hn_pt.shape)
        print(hn_ms, hn_pt)
        assert np.allclose(hn_ms.asnumpy(), hn_pt.detach().numpy(), 1e-3)