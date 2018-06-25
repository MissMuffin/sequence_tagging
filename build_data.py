from model.config import Config
from model.conll_dataset import CoNLLDataset
from model.data_utils import WORD_UNK, WORD_NUM, \
    write_vocab, load_vocab, export_trimmed_embeddings, processing_word, CHAR_NUM, CHAR_UNK, get_embeddings_vocab


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)

    # Build word and tag vocabs
    vocab_words, vocab_tags = CoNLLDataset([config.filename_dev, config.filename_train, config.filename_test],
                                           processing_word(lowercase=True)).get_word_tag_vocabs()

    get_word = lambda line: line.strip().split(' ')[0]
    vocab_glove = get_embeddings_vocab(config.filename_word_embeddings, get_word)
    vocab = vocab_words & vocab_glove | {WORD_UNK, WORD_NUM}

    # Save vocab
    write_vocab(vocab,      config.filename_words)
    write_vocab(vocab_tags, config.filename_tags)

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_embeddings(vocab,
                              get_word,
                              config.filename_word_embeddings,
                              config.filename_word_embeddings_trimmed,
                              config.dim_word)

    # Build and save char vocab
    get_char = lambda line: line[0]
    vocab_chars = get_embeddings_vocab(config.filename_char_embeddings, get_char) | {CHAR_UNK, CHAR_NUM}
    write_vocab(vocab_chars, config.filename_chars)

    vocab_chars = load_vocab(config.filename_chars)
    export_trimmed_embeddings(vocab_chars,
                              get_char,
                              config.filename_char_embeddings,
                              config.filename_char_embeddings_trimmed,
                              config.dim_char)


if __name__ == "__main__":
    main()
