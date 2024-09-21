import numpy as np


class FastText:
    def __init__(self, vocab_size, embedding_dim, ngram_range=3):
        self.vocab_size = vocab_size

    def generate_ngrams(self, word, override_n=None):
        if override_n:
            n = override_n
        else:
            n = self.ngram_range

        word = f'<{word}>'
        ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
        return ngrams
