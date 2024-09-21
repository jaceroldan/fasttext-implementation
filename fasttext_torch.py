import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.target_embeddings = nn.Embedding

class FastText:
    def __init__(self, vocab_size, embedding_dim, ngram_range=3):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ngram_range = ngram_range

    def generate_ngrams(self, word, override_n=None):
        if override_n:
            n = override_n
        else:
            n = self.ngram_range

        word = f'<{word}>'
        ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
        return ngrams
