import nltk
from torch.utils import data


class Bigrammer(data.Dataset):
    def __init__(self, corpus: list):
        """
        :param corpus: list of normalized sentences
        """
        word2idx, idx2word = self.__vocabs__(corpus)
        assert len(word2idx) == len(idx2word)
        self.word2idx = word2idx
        self.idx2word = idx2word

        left = []
        right = []
        for pairs in [nltk.bigrams(c.split()) for c in corpus]:
            for l, r in pairs:
                left.append(self.word2idx[l])
                right.append(self.word2idx[r])
        self.left = left
        self.right = right
        self.half_len = len(self.left)

    def __vocabs__(self, corpus):
        """
        :return: vocabulary {word -> id}
        """
        idx2word = dict(enumerate(set([word for sent in corpus for word in sent.split()])))
        return {v: k for k, v in idx2word.items()}, idx2word

    def __len__(self):
        return 2 * self.half_len

    def __getitem__(self, index):
        """
        Generates one sample of data
        """
        idx = index % self.half_len
        if index // self.half_len == 0:
            return self.left[idx], self.right[idx]
        else:
            return self.right[idx], self.left[idx]
