import unittest
import mindspore
import numpy as np
from rnns.rnn_cells import LSTMCell
from mindspore import Tensor
import torch

# mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
# mindspore.set_context(enable_graph_kernel=True)

class TestGRUCell(unittest.TestCase):
    def setUp(self):
        self.input_size, self.hidden_size = 16, 32
        self.x = np.random.randn(1, self.input_size)
        self.hx = np.random.randn(1, self.hidden_size)
        self.cx = np.random.randn(1, self.hidden_size)

    def test_gru_cell(self):
        net = LSTMCell(self.input_size, self.hidden_size)
        x = Tensor(self.x, mindspore.float32)
        hx = Tensor(self.hx, mindspore.float32)
        cx = Tensor(self.cx, mindspore.float32)
        hn, cn = net(x, (hx, cx))
        assert hn.shape == (1, self.hidden_size)
        assert cn.shape == (1, self.hidden_size)
    
    def test_gru_cell_forward(self):
        net_ms = LSTMCell(self.input_size, self.hidden_size)
        x_ms = Tensor(self.x, mindspore.float32)
        hx_ms = Tensor(self.hx, mindspore.float32)
        cx_ms = Tensor(self.cx, mindspore.float32)
        import time
        hn_ms, cn_ms = net_ms(x_ms, (hx_ms, cx_ms))
        s = time.time()
        hn_ms, cn_ms = net_ms(x_ms, (hx_ms, cx_ms))
        t = time.time()
        ms_cost = t - s
        print(ms_cost)

        net_pt = torch.nn.LSTMCell(self.input_size, self.hidden_size)
        net_pt.weight_ih = torch.nn.Parameter(torch.Tensor(net_ms.weight_ih.asnumpy()))
        net_pt.weight_hh = torch.nn.Parameter(torch.Tensor(net_ms.weight_hh.asnumpy()))
        net_pt.bias_ih = torch.nn.Parameter(torch.Tensor(net_ms.bias_ih.asnumpy()))
        net_pt.bias_hh = torch.nn.Parameter(torch.Tensor(net_ms.bias_hh.asnumpy()))

        x_pt = torch.tensor(self.x, dtype=torch.float32)
        hx_pt = torch.tensor(self.hx, dtype=torch.float32)
        cx_pt = torch.tensor(self.cx, dtype=torch.float32)
        hn_pt, cn_pt = net_pt(x_pt, (hx_pt, cx_pt))
        s = time.time()
        hn_pt, cn_pt = net_pt(x_pt, (hx_pt, cx_pt))
        t = time.time()
        pt_cost = t - s
        print(pt_cost)

        print(f"ms/pt: {ms_cost/pt_cost}")
        # print(hn_ms.shape, hn_pt.shape)
        # print(hn_ms, hn_pt)
        assert np.allclose(hn_ms.asnumpy(), hn_pt.detach().numpy(), 1e-3)
        assert np.allclose(cn_ms.asnumpy(), cn_pt.detach().numpy(), 1e-3)