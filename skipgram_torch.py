import os

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset


class CustomColeridgeDataset(Dataset):



class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()

        self.W_in = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)
        self.W_out = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)

    def forward(self, center_word_idxs, context_word_idxs):
        center_embeddings = self.W_in[center_word_idxs]
        context_embeddings = self.W_out[context_word_idxs]
        scores = torch.sum(center_embeddings * context_embeddings, dim=1)
        return scores


def skipgram_loss(scores, true_labels):
    loss = nn.BCEWithLogitsLoss()(scores, true_labels)
    return loss


def generate_negative_samples(batch_size, vocab_size, num_neg_samples, true_context_idxs):
    """
    batch_size: number of "center words" per batch
    num_neg_samples: number of negative samples per context word — how much noise
    true_context_idxs: actual context word indices
    """
    negative_samples = torch.randint(0, vocab_size, (batch_size, num_neg_samples))
    for i in range(batch_size):
        while true_context_idxs[i] in negative_samples[i]:
            negative_samples[i] = torch.randint(0, vocab_size, (num_neg_samples,))

    return negative_samples


if __name__ == '__main__':
    vocab_size = 10_000
    embedding_dim = 100
    batch_size = 64
    num_neg_samples = 5
    learning_rate = 0.01
    num_epochs = 10

    for i in range(num_epochs):
        model = SkipGramModel(vocab_size, embedding_dim)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # TODO: Extract this from the dataset of 1000 files
        center_word_idxs = torch.randint(0, vocab_size, (batch_size,))
        context_word_idxs = torch.randint(0, vocab_size, (batch_size,))

        neg_samples = generate_negative_samples(batch_size, vocab_size, num_neg_samples, context_word_idxs)

        true_labels = torch.ones(batch_size)
        neg_labels = torch.zeros(batch_size, num_neg_samples)

        optimizer.zero_grad()
        positive_scores = model(center_word_idxs, context_word_idxs)

        negative_scores = torch.zeros(batch_size, num_neg_samples)
        for i in range(num_neg_samples):
            negative_scores[:, i] = model(center_word_idxs, neg_samples[:, i])
        

        positive_loss = skipgram_loss(positive_scores, true_labels)
        negative_loss = skipgram_loss(negative_scores.view(-1), neg_labels.view(-1))

        total_loss = positive_loss + negative_loss
        total_loss.backward()

        optimizer.step()

        print(f"Loss: {total_loss.item()}")
