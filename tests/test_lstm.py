import unittest
import mindspore
import numpy as np
from mindspore import context, Tensor
from rnns import LSTM

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.input_size, self.hidden_size = 10, 20
        self.x = np.random.randn(3, 10, self.input_size)

    def test_rnn(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, (h, c) = rnn(inputs)

        assert output.shape == (3, 10, 20)
        assert h.shape == (1, 3, 20)
        assert c.shape == (1, 3, 20)

    def test_rnn_bidirection(self):
        rnn = LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, (h, c) = rnn(inputs)

        assert output.shape == (3, 10, 20 * 2)
        assert h.shape == (2, 3, 20)
        assert c.shape == (2, 3, 20)

    def test_rnn_multi_layer(self):
        rnn = LSTM(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, (h, c) = rnn(inputs)

        assert output.shape == (3, 10, 20)
        assert h.shape == (1 * 3, 3, 20)
        assert c.shape == (1 * 3, 3, 20)