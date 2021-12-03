# mindspore-rnns
This repository is an unofficial implementation of RNN operators for MindSpore. In order to bridge the gap between Ascend and GPU, and add the whole support of RNN, GRU and LSTM on CPU.

### How to use
Copy the implementation in your source code. If you want to use RNN opeators, code like below:

```python
import mindspore
from rnns.rnns import RNN, GRU, LSTM

rnn = RNN(16, 32, num_layers=1)
gru = GRU(16, 32, has_bias=False)
lstm = LSTM(16, 32, bidrectional=True)
```

If you want to use RNN Cells to achieve your own modified RNN operator, use code like below:

```python
import mindspore
from rnns.rnn_cells import RNNCell, GRUCell, LSTMCell

rnn_cell = RNNCell(16, 32)
gru_cell = GRUCell(16, 32)
lstm_cell = LSTMCell(16, 32)
```



