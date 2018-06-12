import sys


class CoNLLDataset(object):
    """Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of processed words in a sentence
        tags: list of processed tags in a sentence

    Example:
        ```python
        data = CoNLLDataset(filename)
        for words, tags in data:
            pass
        ```

    """
    def __init__(self, filename, processing_word=lambda word: word, processing_tag=lambda tag: tag, max_sentences=sys.maxsize):
        """
        Args:
            filename: path to the file
            processing_word: (optional) function that takes a word as input
            processing_tag: (optional) function that takes a tag as input
            max_sentences: (optional) max number of sentences to yield

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_sentences = max_sentences
        self.length = None

    def __iter__(self):
        sentences: int = 0
        with open(self.filename) as f:
            words, tags = [], []
            for line in f:
                line = line.strip()
                if line and not line.startswith("-DOCSTART-"):
                    ls = line.split(' ')
                    words.append(self.processing_word(ls[0]))
                    tags.append(self.processing_tag(ls[-1]))
                elif words:
                    sentences += 1
                    if sentences > self.max_sentences:
                        break
                    yield words, tags
                    words, tags = [], []

    def __len__(self):
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = sum(1 for _ in self)
        return self.length

    def get_char_vocab(self):
        """Build char vocabulary from an iterable of datasets objects

        Returns:
            a set of all the characters in the dataset

        """
        return {char for words, _ in self for word in words for char in word}

    @staticmethod
    def get_vocabs(datasets):
        """Build vocabulary from an iterable of datasets objects

        Args:
            datasets: a list of dataset objects

        Returns:
            a set of all words in the dataset

        """
        print("Building vocab...")
        vocab_words = set()
        vocab_tags = set()
        for dataset in datasets:
            for words, tags in dataset:
                vocab_words.update(words)
                vocab_tags.update(tags)
        print("- done. {} tokens".format(len(vocab_words)))
        return vocab_words, vocab_tags
