import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2Vec(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()
        ## From documentation
        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        # This module is often used to store word embeddings and retrieve them using indices.
        # The input to the module is a list of indices, and the output is the corresponding
        # word embeddings.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        """
        :param x: id of word
        :return: predicted context id in one-hot vector format
        """
        emb = self.embedding(x)
        y_pred = self.classifier(emb)
        return F.log_softmax(y_pred, dim=0)
