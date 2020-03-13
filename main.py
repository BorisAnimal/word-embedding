import torch
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm

from src import dataset_reader
from src.dataset import Bigrammer
from src.word2vec import Word2Vec

"""
    Adopted from tutorial:
    https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
"""


def train(model, dataset,
          embedding_path,
          num_epochs=20,
          lr=0.01,
          batch_size=512,
          device=torch.device('cuda'),
          ):
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True)
    min_loss = None

    for epoch in range(num_epochs):
        loss_val = 0
        for context, target in tqdm(dataloader, desc="Batches"):
            x = context.to(device)
            y_true = torch.LongTensor(target).to(device)
            optim.zero_grad()

            y_pred = model(x)
            loss = F.nll_loss(y_pred, y_true)  # .view(batch_size, -1)
            loss_val += loss.item()
            loss.backward()
            optim.step()
        mean_loss = loss_val / len(dataloader)
        if min_loss is None or mean_loss < min_loss:
            min_loss = mean_loss
            print("Model saved to {}".format(embedding_path))
            torch.save((dataset, model.embedding), embedding_path)
        print(f'Loss at epoch {epoch}: {mean_loss}')


def main(embedding_path="models/embedding.pth",
         emb_dim=64):
    corpus = dataset_reader.main()
    corpus = corpus[::10]  # because full dataset is 6 hours per epoch
    bigrams = Bigrammer(corpus)
    v_size = len(bigrams.word2idx)
    print("Vocabulary size: {}".format(v_size))

    w2v = Word2Vec(v_size, emb_dim)

    train(w2v, bigrams, embedding_path)


if __name__ == '__main__':
    main()
