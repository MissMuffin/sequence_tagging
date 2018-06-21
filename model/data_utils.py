import numpy as np

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building glove vocab from {}...".format(filename))
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} words".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    """
    print("Writing vocab in {}...".format(filename))
    with open(filename, "w") as f:
        f.write("\n".join(vocab))
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        with open(filename) as f:
            return {word.strip(): idx for idx, word in enumerate(f)}
    except IOError:
        raise MyIOError(filename)


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """

    def f(word):
        # 0. get chars of word
        if vocab_chars is not None:
            char_ids = [vocab_chars[char] for char in word if char in vocab_chars]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():  # TODO: Filter decimal, Replace digit by 'D'
            word = NUM

        # 2. get id of word
        if vocab_words is None:
            word_id = word
        else:
            if word in vocab_words:
                word_id = vocab_words[word]
            elif allow_unk:
                word_id = vocab_words[UNK]
            else:
                raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is None:
            return word_id
        else:
            return char_ids, word_id

    return f


def processing_chars_word_id(vocab_chars, vocab_words, lowercase=False, allow_unk=True):
    """Return a lambda function that transform a word (string) into
    a tuple (list, int) of the ids of the characters and the id of the word.

    Args:
        vocab_chars: dict[char] = idx
        vocab_words: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        char_ids = [vocab_chars[char] for char in word if char in vocab_chars]
        word_id = processing_word_id(vocab_words, lowercase, allow_unk)(word)
        return char_ids, word_id

    return f


def processing_word_id(vocab_words, lowercase=False, allow_unk=True):
    """Return a lambda function that transform a word (string) into its id.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = 12345
                 = word id
    """
    def f(word):
        word = processing_word(lowercase)(word)
        if word in vocab_words:
            return vocab_words[word]
        if allow_unk:
            return vocab_words[UNK]
        raise Exception("Unknown key is not allowed. Check that your vocab (tags?) is correct")

    return f


def processing_word(lowercase=False):
    """Return a lambda function that processes a word (string).

    Returns:
        The a lambda function that processes a word (string).
    """
    def f(word):
        if lowercase:
            word = word.lower()
        if word.isdigit():  # TODO: Filter decimal, Replace digit by 'D'
            word = NUM
        return word

    return f

def pad_words(words_ids, pad_tok=0):
    """
    Args:
        words_ids: the sentences with ids of words
        pad_tok: the char to pad with

    Returns:
        a list of lists where each sublist has same length

    """
    max_length = max(map(lambda x: len(x), words_ids))
    return _pad_sequences(words_ids, pad_tok, max_length)


def pad_chars(charss_ids, pad_tok=0):
    """
    Args:
        charss_ids: the sentences with ids of chars
        pad_tok: the char to pad with

    Returns:
        a list of lists of lists where each sublist has same length

    """
    max_length_word = max([max(map(lambda x: len(x), char_ids)) for char_ids in charss_ids])
    sequence_padded, sequence_length = [], []
    for chars_ids in charss_ids:
        # all words are same length now
        sp, sl = _pad_sequences(chars_ids, pad_tok, max_length_word)
        sequence_padded.append(sp)
        sequence_length.append(sl)

    max_length_sentence = max(map(lambda x: len(x), charss_ids))
    sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
    sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    return sequence_padded, sequence_length


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded.append(seq_)
        sequence_length.append(min(len(seq), max_length))

    return sequence_padded, sequence_length


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
