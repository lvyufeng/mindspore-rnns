import unittest
import mindspore
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
from rnns import RNN
import torch

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.input_size, self.hidden_size = 10, 20
        self.x = np.random.randn(3, 10, self.input_size)

    def test_rnn(self):
        rnn = RNN(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 20)
        assert h.shape == (1, 3, 20)

    def test_rnn_bidirection(self):
        rnn = RNN(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 20 * 2)
        assert h.shape == (2, 3, 20)

    def test_rnn_multi_layer(self):
        rnn = RNN(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 20)
        assert h.shape == (1 * 3, 3, 20)

    def test_forward_cmp(self):
        # mindspore rnn
        rnn_ms = RNN(self.input_size, self.hidden_size, batch_first=True)
        inputs_ms = Tensor(self.x, mindspore.float32)

        # pytorch rnn
        rnn_pt = torch.nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        inputs_pt = torch.Tensor(self.x)

        # set mindspore parameters to pytorch
        for param in rnn_ms.w_ih_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))
        for param in rnn_ms.w_hh_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))        
        for param in rnn_ms.b_ih_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))
        for param in rnn_ms.b_hh_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))        

        # forward
        outputs_ms, h_ms = rnn_ms(inputs_ms)
        outputs_pt, h_pt = rnn_pt(inputs_pt)

        assert np.allclose(outputs_ms.asnumpy(), outputs_pt.detach().numpy(), 1e-4, 1e-4)
        assert np.allclose(h_ms.asnumpy(), h_pt.detach().numpy(), 1e-4, 1e-4)

    def test_backward_cmp(self):
        # mindspore rnn
        rnn_ms = RNN(self.input_size, self.hidden_size, batch_first=True)
        inputs_ms = Tensor(self.x, mindspore.float32)

        # pytorch rnn
        rnn_pt = torch.nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        inputs_pt = torch.Tensor(self.x)

        # set mindspore parameters to pytorch
        for param in rnn_ms.w_ih_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))
        for param in rnn_ms.w_hh_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))        
        for param in rnn_ms.b_ih_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))
        for param in rnn_ms.b_hh_list:
            setattr(rnn_pt, param.name, torch.nn.Parameter(torch.Tensor(param.asnumpy())))        

        # forward
        outputs_ms, h_ms = rnn_ms(inputs_ms)
        outputs_pt, h_pt = rnn_pt(inputs_pt)

        assert np.allclose(outputs_ms.asnumpy(), outputs_pt.detach().numpy(), 1e-4, 1e-4)
        assert np.allclose(h_ms.asnumpy(), h_pt.detach().numpy(), 1e-4, 1e-4)

        # backward
        grad_param = ops.GradOperation(get_by_list=True)
        rnn_ms_grads = grad_param(rnn_ms, rnn_ms.trainable_params())(inputs_ms)

        outputs_pt.backward(torch.ones_like(outputs_pt), retain_graph=True)
        h_pt.backward(torch.ones_like(h_pt), retain_graph=True)
        rnn_pt_grads = [param.grad for param in rnn_pt.parameters()]
        
        for ms_grad, pt_grad in zip(rnn_ms_grads, rnn_pt_grads):
            print(ms_grad.shape)
            print(pt_grad.shape)
            assert np.allclose(ms_grad.asnumpy(), pt_grad.detach().numpy(), 1e-4, 1e-4)