import os
import sys

from model.conll_dataset import CoNLLDataset
from .data_utils import get_trimmed_embeddings, load_vocab, processing_chars_word_id, \
    processing_word_id
from .general_utils import get_logger


class Config:

    # general config
    dir_output = "results/test/"
    dir_model = dir_output + "model.weights/"
    logger = get_logger(dir_output + "log.txt")

    dim_word = 300
    dim_char = 300
    use_pretrained_words = True
    use_pretrained_chars = True

    # training
    train_word_embeddings = False
    train_char_embeddings = True
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1  # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_lstm    = 300   # lstm on words
    hidden_size_char    = 100   # lstm on chars
    features_per_ngram  = 25    # cnn on char embeddings
    max_size_ngram      = 6     # cnn on char embeddings
    highway_layers      = 2     # highway network

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True      # if crf, training is 1.7x slower on CPU
    use_chars = True    # if char embedding, training is 3.5x slower on CPU

    def __init__(self, load=True,
                 dim_word=dim_word,
                 dim_char=dim_char,
                 use_pretrained_words=use_pretrained_words,
                 use_pretrained_chars=use_pretrained_chars):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.dim_word = dim_word
        self.dim_char = dim_char
        self.use_pretrained_words = use_pretrained_words
        self.use_pretrained_chars = use_pretrained_chars

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        self.filename_word_embeddings = "data/wiki.de.vec"
        self.filename_word_embeddings_trimmed = "data/wiki.de.vec.trimmed.npz"

        self.filename_char_embeddings = "data/char-embeddings-{}d.txt".format(dim_char)
        self.filename_char_embeddings_trimmed = "data/char-embeddings-{}d-trimmed.npz".format(dim_char)

        self.filename_dev = "data/germeval2014/NER-de-dev-CoNLL2003.txt"
        self.filename_test = "data/germeval2014/NER-de-test-CoNLL2003.txt"
        self.filename_train = "data/germeval2014/NER-de-train-CoNLL2003.txt"

        self.max_iter = sys.maxsize  # max number of examples in a dataset

        # vocabs (created from dataset with build_data.py)
        self.filename_words = "data/words.txt"
        self.filename_tags = "data/tags.txt"
        self.filename_chars = "data/chars-{}d.txt".format(dim_char)

        if load:
            """Loads vocabulary, processing functions, embeddings and datasets

            Supposes that build_data.py has been run successfully and that
            the corresponding files have been created (vocab and trimmed GloVe
            vectors)

            """
            # 1. vocabulary
            self.vocab_chars = load_vocab(self.filename_chars)
            self.vocab_words = load_vocab(self.filename_words)
            self.vocab_tags  = load_vocab(self.filename_tags)

            # 2. get processing functions that map str -> id
            if self.use_chars:
                self.processing_word = processing_chars_word_id(self.vocab_chars, self.vocab_words, lowercase=True, allow_unk=True)
            else:
                self.processing_word = processing_word_id(self.vocab_words, lowercase=True, allow_unk=True)

            self.processing_tag = processing_word_id(self.vocab_tags, lowercase=False, allow_unk=False)

            # 3. get pre-trained embeddings
            self.word_embeddings = get_trimmed_embeddings(self.filename_word_embeddings_trimmed) if self.use_pretrained_words else None
            self.char_embeddings = get_trimmed_embeddings(self.filename_char_embeddings_trimmed) if self.use_pretrained_chars else None

            # 4. get datasets
            self.dataset_dev = CoNLLDataset(self.filename_dev, self.processing_word, self.processing_tag, self.max_iter)
            self.dataset_train = CoNLLDataset(self.filename_train, self.processing_word, self.processing_tag, self.max_iter)
            self.dataset_test = CoNLLDataset(self.filename_test, self.processing_word, self.processing_tag, self.max_iter)
