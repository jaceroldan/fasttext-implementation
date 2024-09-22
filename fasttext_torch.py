import os
import json

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils

from torch.utils.data import DataLoader

from skipgram_torch import Vocabulary, CustomColeridgeDataset, skipgram_loss


class FastTextVocabulary(Vocabulary):
    def __init__(self, min_freq=1, ngram_range=(3, 6)):
        super().__init__(min_freq)
        self.ngram_range = ngram_range
        self.ngram2idx = {}
        self.idx2ngram = {}
        self.ngram_count = 1 # as with the old version, we start at 1, and leave 0 for <UNK>

    def _get_ngrams(self, word):
        ngrams = []
        word = f'<{word}>'
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngrams.extend([word[i:i+n] for i in range(len(word) - n + 1)])
        return ngrams
    
    def build_vocab(self, texts):
        super().build_vocab(texts)

        for word in self.word2idx:
            ngrams = self._get_ngrams(word)
            for ngram in ngrams:
                if ngram not in self.ngram2idx:
                    self.ngram2idx[ngram] = self.ngram_count
                    self.idx2ngram[self.ngram_count] = ngram
                    self.ngram_count += 1
    
    def encode_word(self, word):
        word_idx = self.word2idx.get(word, 0)
        ngram_idxs = [self.ngram2idx.get(ng, 0) for ng in self._get_ngrams(word)]
        return word_idx, ngram_idxs
    
    def ngram_vocab_size(self):
        return len(self.ngram2idx)
    


class FastTextColeridgeDataset(CustomColeridgeDataset):
    def __getitem__(self, idx):
        train_id = self.train_items.iloc[idx]['Id']
        curr_path = os.path.join(self.json_dir, train_id + '.json')
        with open(curr_path, 'r') as file:
            curr_json = json.load(file)

        text = ''.join([cj['text'] for cj in curr_json])
        word_indices = self.vocab.encode(text)
        
        center_context_pairs = []
        for i, center_word_idx in enumerate(word_indices):
            center_word, center_ngrams = self.vocab.encode_word(center_word_idx)
            for j in range(max(0, i - self.window_size), min(len(word_indices), i + self.window_size + 1)):
                if i != j:
                    context_word, context_ngrams = self.vocab.encode_word(word_indices[j])
                    center_context_pairs.append((center_word, center_ngrams, context_word, context_ngrams))

        return center_context_pairs



class FastTextSkipGramModel(nn.Module):
    def __init__(self, vocab_size, ngram_vocab_size, embedding_dim):
        super(FastTextSkipGramModel, self).__init__()
        self.word_embeddings = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.01)
        self.ngram_embeddings = nn.Parameter(torch.randn(ngram_vocab_size, embedding_dim) * 0.01)

    def forward(self, center_word_idx, center_ngram_idxs, context_word_idx, context_ngram_idxs):
        center_word_embedding = self.word_embeddings[center_word_idx]
        center_ngram_embeddings = torch.sum(self.ngram_embedding[center_ngram_idxs], dim=0)
        center_embedding = center_word_embedding + center_ngram_embeddings

        context_word_embedding = self.word_embeddings[context_word_idx]
        context_ngram_embeddings = torch.sum(self.ngram_embeddings[context_ngram_idxs], dim=0)
        context_embedding = context_word_embedding + context_ngram_embeddings

        score = torch.sum(center_embedding, context_embedding, dim=1)
        return score


if __name__ == '__main__':
    vocab = FastTextVocabulary(min_freq=5)
    
    train = pd.read_csv('./datasets/train.csv')
    train_items = train.sample(n=1000, random_state=42)
    texts = []

    for i in range(len(train_items)):
        curr_path = os.path.join(os.getcwd(), 'datasets', 'train', train_items.iloc[i]['Id'] + '.json')
        with open(curr_path, 'r') as file:
            curr_json = json.load(file)
            texts.append(''.join([cj['text'] for cj in curr_json]))
    
    vocab.build_vocab(texts)

    dataset = FastTextColeridgeDataset(csv_file='./datasets/train.csv', json_dir='./datasets/train/', vocab=vocab)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: [item for sublist in x for item in sublist])

    vocab_size = vocab.vocab_size()
    ngram_vocab_size = vocab.ngram_vocab_size()
    embedding_dim = 100
    model = FastTextSkipGramModel(vocab_size, ngram_vocab_size, embedding_dim)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for batch in dataloader:
        center_word_idxs, center_ngram_idxs, context_word_idxs, context_ngram_idxs = zip(*batch)
        
        center_word_idxs = torch.tensor(center_word_idxs, dtype=torch.long)
        context_word_idxs = torch.tensor(context_word_idxs, dtype=torch.long)

        center_ngram_idxs = [torch.tensor(ngrams, dtype=torch.long) for ngrams in center_ngram_idxs]
        context_ngram_idxs = [torch.tensor(ngrams, dtype=torch.long) for ngrams in context_ngram_idxs]

        center_ngram_idxs_padded = rnn_utils.pad_sequence(center_ngram_idxs, batch_first=True, padding_value=0)
        context_ngram_idxs_padded = rnn_utils.pad_sequence(context_ngram_idxs, batch_first=True, padding_value=0)

        optimizer.zero_grad()
        scores = model(center_word_idxs, center_ngram_idxs, context_word_idxs, context_ngram_idxs)

        true_labels = torch.ones_like(scores)
        loss = skipgram_loss(scores, true_labels)
        loss.backward()
        optimizer.step()

        print(f'Loss: {loss.item()}')

