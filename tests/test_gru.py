import unittest
import mindspore
import numpy as np
from mindspore import Tensor, context
from rnns import GRU

class TestRNN(unittest.TestCase):
    def setUp(self):
        self.input_size, self.hidden_size = 16, 32
        self.x = np.random.randn(3, 10, self.input_size)
        context.set_context(mode=context.PYNATIVE_MODE)

    def test_rnn(self):
        rnn = GRU(self.input_size, self.hidden_size, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32)
        assert h.shape == (1, 3, 32)

    def test_rnn_bidirection(self):
        rnn = GRU(self.input_size, self.hidden_size, batch_first=True, bidirectional=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32 * 2)
        assert h.shape == (2, 3, 32)

    def test_rnn_multi_layer(self):
        rnn = GRU(self.input_size, self.hidden_size, num_layers=3, batch_first=True)
        inputs = Tensor(self.x, mindspore.float32)
        output, h = rnn(inputs)

        assert output.shape == (3, 10, 32)
        assert h.shape == (1 * 3, 3, 32)