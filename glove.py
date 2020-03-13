import os
import random
from datetime import datetime

import nltk
import numpy as np
from keras import backend as K
from keras.layers import Embedding, Input, Add, Dot, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import skipgrams
from scipy.sparse import lil_matrix
from scipy.sparse import save_npz, load_npz

from src import dataset_reader


## Taken from tutorial:
# https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010
## With open source:
# https://github.com/thushv89/exercises_thushv_dot_com/blob/master/glove_light_on_math_ml/glove_light_on_math_ml.ipynb


def generate_cooc_matrix(text, tokenizer, window_size, n_vocab, use_weighting=True):
    sequences = tokenizer.texts_to_sequences(text)

    cooc_mat = lil_matrix((n_vocab, n_vocab), dtype=np.float32)
    for sequence in sequences:
        for i, wi in zip(np.arange(window_size, len(sequence) - window_size), sequence[window_size:-window_size]):
            context_window = sequence[i - window_size: i + window_size + 1]
            distances = np.abs(np.arange(-window_size, window_size + 1))
            distances[window_size] = 1.0
            nom = np.ones(shape=(window_size * 2 + 1,), dtype=np.float32)
            nom[window_size] = 0.0

            if use_weighting:
                cooc_mat[wi, context_window] += nom / distances  # Update element
            else:
                cooc_mat[wi, context_window] += nom

    return cooc_mat


def get_model(v_size):
    """
    GloVe
    :param v_size: size of vocabulary
    :return: keras compiled model
    """
    w_i = Input(shape=(1,))
    w_j = Input(shape=(1,))

    emb_i = Flatten()(Embedding(v_size, 96, input_length=1)(w_i))
    emb_j = Flatten()(Embedding(v_size, 96, input_length=1)(w_j))

    ij_dot = Dot(axes=-1)([emb_i, emb_j])

    b_i = Flatten()(
        Embedding(v_size, 1, input_length=1)(w_i)
    )
    b_j = Flatten()(
        Embedding(v_size, 1, input_length=1)(w_j)
    )

    pred = Add()([ij_dot, b_i, b_j])

    def glove_loss(y_true, y_pred):
        return K.sum(
            K.pow((y_true - 1) / 100.0, 0.75) * K.square(y_pred - K.log(y_true))
        )

    model = Model(inputs=[w_i, w_j], outputs=pred)
    model.compile(loss=glove_loss, optimizer=Adam(lr=0.0001))
    return model


# Params
# corpus_path = "data/eng_wikipedia_2007_1M-sentences.txt"
matrix_path = "models/cooc_matrix.npz"

if __name__ == '__main__':
    corpus, tokenizer = dataset_reader.main()
    v_size = dataset_reader.v_size

    print("Creating co-occurence matrix...")
    if os.path.isfile(matrix_path):
        start = datetime.now()
        cooc_mat = generate_cooc_matrix(corpus, tokenizer, 4, v_size, True)
        save_npz(matrix_path, cooc_mat.tocsr())
        print("Matrix ({} by {}) created in {}".format(v_size, v_size, datetime.now() - start))
    else:
        cooc_mat = load_npz(matrix_path).tolil()
        print('Matrix was loaded from disk')

    print("Training model...")
    model = get_model(v_size)

    batch_size = 128
    copy_docs = list(corpus)
    index2word = dict(zip(tokenizer.word_index.values(), tokenizer.word_index.keys()))
    """ Each epoch """
    for ep in range(10):

        # valid_words = get_valid_words(docs, 20, tokenizer)

        random.shuffle(copy_docs)
        losses = []
        """ Each document (i.e. movie plot) """
        for doc in copy_docs:

            seq = tokenizer.texts_to_sequences([doc])[0]

            """ Getting skip-gram data """
            # Negative samples are automatically sampled by tf loss function
            wpairs, labels = skipgrams(
                sequence=seq, vocabulary_size=v_size, negative_samples=0.0, shuffle=True
            )

            if len(wpairs) == 0:
                print("Len (wpairs) == 0")
                continue

            sg_in, sg_out = zip(*wpairs)
            sg_in, sg_out = np.array(sg_in).reshape(-1, 1), np.array(sg_out).reshape(-1, 1)
            x_ij = np.array(cooc_mat[sg_in[:, 0], sg_out[:, 0]]).reshape(-1, 1) + 1

            assert np.all(np.array(labels) == 1)
            assert x_ij.shape[0] == sg_in.shape[0], 'X_ij {} shape does not sg_in {}'.format(x_ij.shape, sg_in.shape)
            """ For each batch in the dataset """
            h = model.fit([sg_in, sg_out], x_ij, batch_size=batch_size, epochs=1, verbose=1)
            print("History: ", h.history)
            l = model.evaluate([sg_in, sg_out], x_ij, batch_size=batch_size, verbose=1)
            losses.append(l)
        print('Loss in epoch {}: {}'.format(ep, np.mean(losses)))
