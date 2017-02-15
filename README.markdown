# RNN from Scratch

Tutorial : [**Unfolding RNNs II** - Vanilla, GRU, LSTM RNNs from scratch in Tensorflow](http://suriyadeepan.github.io/2017-02-13-unfolding-rnn-2/)

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

## Hallucinations

> Our Abinlinution is tend's so, it's much last to zero familization in the high school society Microsoft we could be meetings change of hemorable to start a startup\n that take \n because server-startups machine whatever they have getting economics end-gradea is decisions like the religion a high-methlity, C8N Disps?And the founders who seem no was the\n gain of beil. Or not replacing. It's not stupide than help-baround togethers. The related\n wealth.Be aid it, not is tried along\n job what's what a startup hubs.The web liked an impressively big\n people the way to be some downbar office themselves gradually be: it's ton, you disappoint it seffect the organization\n being the last time when we were always be founded, eithin the last in creating business are raising geats in an adbitious problem is friex, he may make a company have to started out to bringing himself in big treen of users.\n Sundrad mericonds who've never olders were, people want. [15]]4. Since the reason\n\n school.

The full list of results is available [here](results.markdown).
