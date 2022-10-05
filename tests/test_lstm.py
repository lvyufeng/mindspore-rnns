import unittest
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor, ParameterTuple
from rnns import LSTM
import torch

# mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
# mindspore.set_context(enable_graph_kernel=True)
# mindspore.set_context(max_call_depth=10000)

class TestLSTM(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.seq_length = 10
        self.input_size, self.hidden_size = 128, 64
        self.x = np.random.randn(self.batch_size, self.seq_length, self.input_size)

    def test_lstm(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, (h, c) = rnn(inputs)

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size)
        assert h.shape == (1, self.batch_size, self.hidden_size)
        assert c.shape == (1, self.batch_size, self.hidden_size)

    def test_lstm_seq_length(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        seq_length = Tensor(np.random.randint(1, self.seq_length, (self.batch_size,)), mindspore.int64)
        output, (h, c) = rnn(inputs, seq_length=seq_length)

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size)
        assert h.shape == (1, self.batch_size, self.hidden_size)
        assert c.shape == (1, self.batch_size, self.hidden_size)

    def test_lstm_long(self):
        self.x = np.random.randn(16, 200, self.input_size)
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        import time
        output, h = rnn(inputs)
        s = time.time()
        output, h = rnn(inputs)
        t = time.time() - s
        print(t)

        rnn_pt = torch.nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = torch.tensor(self.x).to(torch.float32)
        output, h = rnn_pt(inputs)
        s = time.time()
        output, h = rnn_pt(inputs)
        t = time.time() - s
        print(t)

        assert output.shape == (16, 200, self.hidden_size)
        assert h[0].shape == (1, 16, self.hidden_size)


    def test_lstm_fp16(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float16)
        output, (h, c) = rnn(inputs)

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size)
        assert h.shape == (1, self.batch_size, self.hidden_size)
        assert c.shape == (1, self.batch_size, self.hidden_size)
        assert output.dtype == mindspore.float16

    def test_lstm_with_hx(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        h0 = Tensor(np.zeros((1, self.batch_size, self.hidden_size)), mindspore.float32)
        c0 = Tensor(np.zeros((1, self.batch_size, self.hidden_size)), mindspore.float32)
        output, (h, c) = rnn(inputs, (h0, c0))

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size)
        assert h.shape == (1, self.batch_size, self.hidden_size)
        assert c.shape == (1, self.batch_size, self.hidden_size)

    def test_lstm_with_hx_fp16(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float16)
        h0 = Tensor(np.zeros((1, self.batch_size, self.hidden_size)), mindspore.float16)
        c0 = Tensor(np.zeros((1, self.batch_size, self.hidden_size)), mindspore.float16)
        output, (h, c) = rnn(inputs, (h0, c0))

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size)
        assert h.shape == (1, self.batch_size, self.hidden_size)
        assert c.shape == (1, self.batch_size, self.hidden_size)

    def test_lstm_bidirection(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, (h, c) = rnn(inputs)

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size * 2)
        assert h.shape == (2, self.batch_size, self.hidden_size)
        assert c.shape == (2, self.batch_size, self.hidden_size)

    def test_lstm_multi_layer(self):
        rnn = LSTM(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, (h, c) = rnn(inputs)

        assert output.shape == (self.batch_size, self.seq_length, self.hidden_size)
        assert h.shape == (1 * 3, self.batch_size, self.hidden_size)
        assert c.shape == (1 * 3, self.batch_size, self.hidden_size)

    def test_forward_cmp(self):
        # mindspore rnn
        rnn_ms = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs_ms = Tensor(self.x, mindspore.float32)

        # pytorch rnn
        rnn_pt = torch.nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
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
        outputs_ms, (h_ms, c_ms) = rnn_ms(inputs_ms)
        outputs_pt, (h_pt, c_pt) = rnn_pt(inputs_pt)
        # print(h_ms.shape, h_pt.shape)
        # # print(h_ms, h_pt)
        # print(outputs_ms, outputs_pt)
        # print(h_ms, h_pt)
        # print(c_ms, c_pt)
        assert np.allclose(outputs_ms.asnumpy(), outputs_pt.detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(h_ms.asnumpy(), h_pt.detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(c_ms.asnumpy(), c_pt.detach().numpy(), 1e-3, 1e-3)

    def test_backward_cmp(self):
        # mindspore rnn
        rnn_ms = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs_ms = Tensor(self.x, mindspore.float32)

        # pytorch rnn
        rnn_pt = torch.nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
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
        outputs_ms, (h_ms, c_ms) = rnn_ms(inputs_ms)
        import time
        ms_s = time.time()
        outputs_ms, (h_ms, c_ms) = rnn_ms(inputs_ms)
        ms_t = time.time() - ms_s
        outputs_pt, (h_pt, c_pt) = rnn_pt(inputs_pt)
        pt_s = time.time()
        outputs_pt, (h_pt, c_pt) = rnn_pt(inputs_pt)
        pt_t = time.time() - pt_s

        print("mindspore:", ms_t)
        print("pytorch:", pt_t)
        assert np.allclose(outputs_ms.asnumpy(), outputs_pt.detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(h_ms.asnumpy(), h_pt.detach().numpy(), 1e-3, 1e-3)
        assert np.allclose(c_ms.asnumpy(), c_pt.detach().numpy(), 1e-3, 1e-3)

        # backward
        grad_param = ops.GradOperation(get_by_list=True)
        rnn_ms_grads = grad_param(rnn_ms, ParameterTuple(rnn_ms.trainable_params()))(inputs_ms)

        outputs_pt.backward(torch.ones_like(outputs_pt), retain_graph=True)
        h_pt.backward(torch.ones_like(h_pt), retain_graph=True)
        c_pt.backward(torch.ones_like(c_pt), retain_graph=True)
        rnn_pt_grads = [param.grad for param in rnn_pt.parameters()]
        
        for ms_grad, pt_grad in zip(rnn_ms_grads, rnn_pt_grads):
            assert np.mean(ms_grad.asnumpy() - pt_grad.detach().numpy()) < 1e-3
