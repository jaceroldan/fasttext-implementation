import json
import os
import re

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=1):
        self.word2idx = {'<UNK>': 0}  # Add a special token for unknown words
        self.idx2word = {0: '<UNK>'}
        self.min_freq = min_freq
    
    def build_vocab(self, texts):
        word_counts = Counter(re.findall(r'\w+', ' '.join(texts).lower()))

        idx = 1  # Start indices from 1 since 0 is reserved for <UNK>
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text):
        # Return word index or 0 if word is not found in vocab
        # TODO: There must be a smarter way to deal with words not in vocabulary.
        # Using the <UNK> token above gets me by for now.
        return [self.word2idx.get(word, 0) for word in re.findall(r'\w+', text.lower())]
    
    def vocab_size(self):
        return len(self.word2idx)


class CustomColeridgeDataset(Dataset):
    def __init__(self, csv_file, json_dir, vocab, window_size=2, n_samples=1000, random_state=42):
        self.train = pd.read_csv(csv_file)
        self.train_items = self.train.sample(n=n_samples, random_state=random_state)
        self.json_dir = json_dir
        self.vocab = vocab
        self.window_size = window_size

    def __len__(self):
        return len(self.train_items)
    
    def __getitem__(self, idx):
        train_id = self.train_items.iloc[idx]['Id']
        curr_path = os.path.join(self.json_dir, train_id + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)

        text = ''.join([cj['text'] for cj in curr_json])
        word_indices = self.vocab.encode(text)

        # generate center-context pairs using the window size
        center_context_pairs = []
        for i, center_word_idx in enumerate(word_indices):
            for j in range(max(0, i - self.window_size), min(len(word_indices), i + self.window_size + 1)):
                if i != j:
                    context_word_idx = word_indices[j]
                    center_context_pairs.append((center_word_idx, context_word_idx))

        return center_context_pairs


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
    vocab = Vocabulary(min_freq=5)

    train = pd.read_csv('./datasets/train.csv')
    train_items = train.sample(n=1000, random_state=42)
    texts = []

    for i in range(len(train_items)):
        curr_path = os.path.join(
            os.getcwd(), 'datasets', 'train', train_items.iloc[i]['Id'] + '.json')

        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            texts.append(''.join([cj['text'] for cj in curr_json]))

    vocab.build_vocab(texts)
    
    dataset = CustomColeridgeDataset(csv_file='./datasets/train.csv', json_dir='./datasets/train/', vocab=vocab)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: [item for sublist in x for item in sublist])

    vocab_size = vocab.vocab_size()
    embedding_dim = 100
    model = SkipGramModel(vocab_size, embedding_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for batch in dataloader:
        center_word_idxs, context_word_idxs = zip(*batch)
        center_word_idxs = torch.tensor(center_word_idxs, dtype=torch.long)
        context_word_idxs = torch.tensor(context_word_idxs, dtype=torch.long)

        if center_word_idxs.max().item() >= vocab_size or context_word_idxs.max().item() >= vocab_size:
            print(f"Error: found out-of-bounds index. Max center: {center_word_idxs.max()}, Max context: {context_word_idxs.max()}")
            continue

        optimizer.zero_grad()
        scores = model(center_word_idxs, context_word_idxs)

        true_labels = torch.ones_like(scores)
        loss = skipgram_loss(scores, true_labels)
        loss.backward()
        optimizer.step()

        # Benchmark loss scores at: 0.02
        print(f'Loss: {loss.item()}')
