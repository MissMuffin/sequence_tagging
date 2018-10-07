from collections import Counter

import numpy as np

glove_dim = 300
glove_token = '840B'
glove_path = 'glove.{0}/glove.{0}.{1}d.txt'.format(glove_token, glove_dim)
char_path = 'char-embeddings-{}d.txt'.format(glove_dim)

char_counter = Counter()
char_vectors = {}

with open(glove_path, 'r', encoding='utf-8') as f:
    for l, line in enumerate(f):
        if l % 10000 == 0:
            print("Line {}".format(l))
        line_split = line.strip().split(' ')
        word = line_split[0]
        vec = np.array(line_split[1:], dtype=float)
        if vec.shape[0] == glove_dim:
            for char in word:
                if ord(char) <= 255:
                    char_counter[char] += 1
                    char_vectors[char] = vec if char not in char_vectors else char_vectors[char] + vec

with open(char_path, 'w', encoding='utf-8') as f:
    for char in char_vectors:
        avg = np.round(char_vectors[char] / char_counter[char], 6).tolist()
        f.write(char + ' ' + ' '.join(str(x) for x in avg) + '\n')
