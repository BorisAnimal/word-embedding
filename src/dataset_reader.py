import os
import pickle
from datetime import datetime

import nltk
from keras.preprocessing.text import Tokenizer


def save(obj, file):
    pickle.dump(obj, open(file, "wb"))


def load(file):
    return pickle.load(open(file, "rb"))


## Taken from tutorial:
# https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010
## With open source:
# https://github.com/thushv89/exercises_thushv_dot_com/blob/master/glove_light_on_math_ml/glove_light_on_math_ml.ipynb

def get_clean_corpus(raw_corpus: list):  # , reference_pattern=r"\[[0-9]+\]"):
    """
    :param raw_corpus: list of raw strings - each begins with number.
    :return: generator
    """
    for i in raw_corpus:
        yield " ".join(i.split()[1:])  # .replace(reference_pattern, "")


def sentence2words_preprocessing(sentence: str, tokenizer=nltk.RegexpTokenizer(r"\w+"), to_lower=True,
                                 keep_numbers=False):
    """
    :return: list of normalized words
    """
    words = tokenizer.tokenize(sentence.lower() if to_lower else sentence)
    if keep_numbers:
        return words
    else:
        return [word for word in words if word.isalpha()]


def load_data(dataset_preprocessed, tokenizer_path):
    tokenizer = load(tokenizer_path)
    with open(dataset_preprocessed, 'r', encoding='utf-8') as d:
        corpus = d.readlines()
    return corpus, tokenizer


def preprocessing(corpus_path, dataset_preprocessed_path, tokenizer_path):
    start = datetime.now()
    # Read raw dataset
    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_corpus = f.readlines()
    print("Lines in dataset: {}".format(len(raw_corpus)))
    # Preporcess raw corpus to cleaned corpus
    corpus = get_clean_corpus(raw_corpus)
    corpus = [" ".join(sentence2words_preprocessing(words)) for words in corpus]
    with open(dataset_preprocessed_path, 'w', encoding='utf-8') as d:
        d.write("\n".join(corpus))
    # Feed corpus to tokenizer
    tokenizer = Tokenizer(num_words=v_size, oov_token='UNK')
    tokenizer.fit_on_texts(corpus)
    save(tokenizer, tokenizer_path)

    print("Data processed in {} sec".format(datetime.now() - start))
    return corpus, tokenizer


# Params
v_size = 50000
corpus_path = "data/eng_wikipedia_2007_1M-sentences.txt"
tokenizer_path = "data/tokenizer.pkl"
dataset_preprocessed_path = "data/clean_corpus.txt"


def main():
    if os.path.exists(tokenizer_path) and os.path.exists(dataset_preprocessed_path):
        corpus, tokenizer = load_data(dataset_preprocessed_path, tokenizer_path)
        print('Existed dataset was loaded')
    else:
        print("Processing dataset...")
        corpus, tokenizer = preprocessing(corpus_path, dataset_preprocessed_path, tokenizer_path)
    return corpus, tokenizer


if __name__ == '__main__':
    main()
