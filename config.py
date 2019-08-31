import os
model_dir = 'models'


class Config:
    model = os.path.join(model_dir, 'output_graph.pbmm')
    alphabet = os.path.join(model_dir, 'models/alphabet.txt')
    lm = os.path.join(model_dir, '/lm.binary')
    trie = os.path.join(model_dir, 'trie')
    extended = False
