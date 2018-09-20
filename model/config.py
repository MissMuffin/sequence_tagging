import os
import sys

from model.conll_dataset import CoNLLDataset
from .data_utils import get_trimmed_glove_vectors, load_vocab, processing_chars_word_id, \
    processing_word_id, processing_word_id_lm, get_trimmed_lm_vectors
from .general_utils import get_logger


class Config:

    # general config
    dir_output = "results/test/"
    dir_model = dir_output + "model.weights/"
    path_log = dir_output + "log.txt"

    # embeddings
    dim_word_lm = 300
    dim_word = 300
    dim_char = 100

    # glove files
    filename_glove = "data/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed_glove = "data/glove.6B.{}d.trimmed.npz".format(dim_word)

    # 1 billion words language model embeddings
    # already trimmed to conll vocab length and content
    filename_trimmed_lm = "data/lm1b_emb_trimmed_d{}.npz".format(dim_word_lm)

    # use pretrained glove embeddings
    use_pretrained_glove = True
    # use pretrained embeddings from 1 billion words language model 
    use_pretrained_lm = True

    # dataset
    filename_dev = "data/coNLL/eng/clean/eng.testa.clean.iob"
    filename_test = "data/coNLL/eng/clean/eng.testb.clean.iob"
    filename_train = "data/coNLL/eng/clean/eng.train.clean.iob"

    max_iter = sys.maxsize  # max number of examples in a dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # vocab from 1 billion words language model embeddings
    # filename_lm_words = "data/language_model/filtered_vocab.txt"

    # training
    train_embeddings = False
    train_embeddings_lm = False
    nepochs          = 15
    dropout          = 0.5
    batch_size       = 20
    lr_method        = "adam"
    lr               = 0.001
    lr_decay         = 0.9
    clip             = -1  # if negative, no clipping
    nepoch_no_imprv  = 3

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 300  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True      # if crf, training is 1.7x slower on CPU
    use_chars = True    # if char embedding, training is 3.5x slower on CPU

    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # log config for run
        self.logger.info("glove word embeddings")
        self.logger.info(">> d{}\t\tpretrained = {}\ttrainable = {}".format(self.dim_word, self.use_pretrained_glove, self.train_embeddings))
        
        self.logger.info("language model word embeddings")
        self.logger.info(">> d{}\t\tpretrained = {}\ttrainable = {}".format(self.dim_word_lm, self.use_pretrained_lm, self.train_embeddings_lm))
        
        self.logger.info("init chars d{}:\t{}".format(self.dim_char, self.use_chars))
        self.logger.info("using CRF:\t{}".format(self.use_crf))
        self.logger.info("\n\n")

        # load if requested (default)
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
            # self.vocab_words_lm = load_vocab(self.filename_lm_words)

            # 2. get processing functions that map str -> id
            if self.use_chars:
                self.processing_word = processing_chars_word_id(self.vocab_chars, self.vocab_words, lowercase=True, allow_unk=True)
            else:
                self.processing_word = processing_word_id(self.vocab_words, lowercase=True, allow_unk=True)
            # self.processing_word_lm = processing_word_id_lm(self.vocab_words_lm, lowercase=False, allow_unk=True)
            self.processing_tag = processing_word_id(self.vocab_tags, lowercase=False, allow_unk=False)

            # 3. get pre-trained embeddings
            self.embeddings_glove = get_trimmed_glove_vectors(self.filename_trimmed_glove) if self.use_pretrained_glove else None
            self.embeddings_lm = get_trimmed_lm_vectors(self.filename_trimmed_lm if self.use_pretrained_lm else None)

            # 4. get datasets
            self.dataset_dev = CoNLLDataset(self.filename_dev, self.processing_word, self.processing_tag, self.max_iter)
            self.dataset_train = CoNLLDataset(self.filename_train, self.processing_word, self.processing_tag, self.max_iter)
            self.dataset_test = CoNLLDataset(self.filename_test, self.processing_word, self.processing_tag, self.max_iter)
