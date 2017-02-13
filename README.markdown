# RNN from Scratch

The objective is to build and train RNNs for dummy tasks, using Tensorflow's **scan** module.

- [x] [Vanilla RNN](/vanilla.py)
- [x] GRU
	- [x] [Single layer GRU](/gru.py)
	- [x] [Stacked GRU](/gru-stacked.py)
- [x] LSTM
	- [x] [Single layer LSTM](/lstm.py)
	- [x] [Stacked LSTM](/lstm-stacked.py)

## Help

```bash
# set path to PAULG_PATH
#  set filename to PAULG_FILENAME
python3 data.py 
# set path to 'data/paulg/' in data.load_data
python3 lstm-stacked.py -t # train
python3 lstm-stacked.py -g --num_words 1000 # generate
```
