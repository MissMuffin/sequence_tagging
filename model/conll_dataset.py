import sys
from typing import Set, Tuple, List, Generator, Callable, Union, TypeVar


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
    T = TypeVar('T')

    def __init__(self, filenames: Union[str, List[str]],
                 processing_word: Callable[[str], T] = lambda word: word,
                 processing_tag: Callable[[str], T] = lambda tag: tag,
                 max_sentences: int = sys.maxsize):
        """
        Args:
            filenames: a single or multiple paths to the files
            processing_word: (optional) function that takes a word as input
            processing_tag: (optional) function that takes a tag as input
            max_sentences: (optional) max number of sentences to yield

        """
        self.filenames = filenames if isinstance(filenames, list) else [filenames]
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_sentences = max_sentences
        self.length = None

    def __iter__(self):
        sentences = 0
        words, tags = [], []
        for filename in self.filenames:
            with open(filename) as f:
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

    def __len__(self) -> int:
        """Iterates once over the corpus to set and store length"""
        if self.length is None:
            self.length = sum(1 for _ in self)
        return self.length

    def get_char_vocab(self) -> Set[T]:
        """Build char vocabulary

        Returns:
            a set of characters in the dataset

        """
        print("Building char vocab from {}...".format(self.filenames))
        vocab_chars = {char for words, tags in self for word in words for char in word}
        print("-done. {} chars".format(len(vocab_chars)))
        return vocab_chars

    def get_word_tag_vocabs(self) -> Tuple[Set[T], Set[T]]:
        """Build words and tags vocabularies

        Returns:
            two sets of words and tags

        """
        print("Building word and tag vocab from {}...".format(self.filenames))
        vocab_words = set()
        vocab_tags = set()
        for words, tags in self:
            vocab_words.update(words)
            vocab_tags.update(tags)
        print("- done. {} words and {} tags".format(len(vocab_words), len(vocab_tags)))
        return vocab_words, vocab_tags

    def get_minibatches(self, minibatch_size: int) -> Generator[Tuple[List[T], List[T]], None, None]:
        """
        Args:
            minibatch_size: (int)

        Yields:
            list of tuples (words, tags)
        """
        words_batch, tags_batch = [], []
        for words, tags in self:
            if len(words_batch) == minibatch_size:
                yield words_batch, tags_batch
                words_batch, tags_batch = [], []

            if type(words[0]) == tuple:
                words = zip(*words)

            words_batch.append(words)
            tags_batch.append(tags)

        if len(words_batch) != 0:
            yield words_batch, tags_batch
