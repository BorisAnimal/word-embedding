GloVe (glove.py) is too slow due to unoptimal co-occurence matrix building,
so decided to implement usual word2vec (w2v.py).
Main concept of word2vec is ...

Dataset used is wikipedia-2007-1M

metric (BATS) taken from [source](https://vecto.readthedocs.io/en/docs/tutorial/evaluating.html#word-analogy-task).
It works as follow: ... 

* почитал аналогии BATS и подумал, что они не очень явные. Даже наоборот.
Поэтому использую обычный google-analogy dataset.
Всё брал [отсюда](https://github.com/vecto-ai/word-benchmarks/blob/9b61d0e067e2e68af22160d09450f82d9544cd70/word-analogy/monolingual/en/)

![Strange analogies in BATS](https://imgur.com/DHVJfxQ)

My results are ...
