import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from model.conll_dataset import CoNLLDataset
from .data_utils import get_chunks, pad_words, pad_chars
from .general_utils import Progbar


class NERModel:
    """Specialized class of Model for NER"""

    def __init__(self, config):
        self.config = config
        self.logger = config.logger

        """Define placeholders = entries to computational graph"""
        batch_size, sentence_length, word_length = None, None, None
        self.word_ids = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[batch_size], name="sequence_lengths")
        self.char_ids = tf.placeholder(tf.int32, shape=[batch_size, sentence_length, word_length], name="char_ids")
        self.word_lengths = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="word_lengths")
        self.labels = tf.placeholder(tf.int32, shape=[batch_size, sentence_length], name="labels")
        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

        """
        Defines self.word_embeddings
        If self.config.word_embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.word_embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[len(self.config.vocab_words), self.config.dim_word])
            else:
                _word_embeddings = tf.Variable(
                    initial_value=self.config.word_embeddings,
                    trainable=self.config.train_word_embeddings,
                    name="_word_embeddings",
                    dtype=tf.float32)
            word_embeddings = tf.nn.embedding_lookup(params=_word_embeddings, ids=self.word_ids,
                                                     name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                if self.config.char_embeddings is None:
                    self.logger.info("WARNING: randomly initializing char vectors")
                    _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[len(self.config.vocab_chars), self.config.dim_char])
                else:
                    _char_embeddings = tf.Variable(
                        initial_value=self.config.char_embeddings,
                        trainable=self.config.train_char_embeddings,
                        name="_char_embeddings",
                        dtype=tf.float32)
                char_embeddings = tf.nn.embedding_lookup(params=_char_embeddings, ids=self.char_ids,
                                                         name="char_embeddings")

                with tf.variable_scope('char-rnn'):
                    # put the time dimension on axis=1
                    # bi lstm on chars
                    batch_size, sentence_length, word_length, char_dim = shapes(char_embeddings)
                    _outputs, _output_states = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=tf.contrib.rnn.LSTMCell(num_units=self.config.hidden_size_char, state_is_tuple=True),
                        cell_bw=tf.contrib.rnn.LSTMCell(num_units=self.config.hidden_size_char, state_is_tuple=True),
                        inputs=tf.reshape(char_embeddings, shape=[batch_size * sentence_length, word_length, self.config.dim_char]),
                        sequence_length=tf.reshape(self.word_lengths, shape=[batch_size * sentence_length]),
                        dtype=tf.float32)

                    # read and concat output
                    _output_state_fw, _output_state_bw = _output_states
                    _, output_fw = _output_state_fw
                    _, output_bw = _output_state_bw

                    char_rnn_output = tf.reshape(tensor=tf.concat([output_fw, output_bw], axis=-1),
                                        shape=[batch_size, sentence_length, 2 * self.config.hidden_size_char])
                    char_rnn_output = tf.nn.dropout(char_rnn_output, self.dropout)

                with tf.variable_scope('char-cnn'):
                    # tf.nn.conv2d expects a tensor of shape [batch, in_height, in_width, in_channels]
                    batch_size, sentence_length, word_length, char_dim = shapes(char_embeddings)
                    char_embeddings_conv = tf.reshape(char_embeddings, [batch_size * sentence_length, word_length, char_dim, 1])

                    filter_heights_widths_features = [(ngram, self.config.dim_char, ngram * self.config.features_per_ngram)
                                                      for ngram in range(2, self.config.max_size_ngram + 1)]
                    sum_features = sum(map(lambda x: x[-1], filter_heights_widths_features))
                    pools = []
                    for height, width, features in filter_heights_widths_features:
                        with tf.variable_scope('conv-maxpool-{}x{}'.format(height, width)):
                            W = tf.get_variable(name='W', shape=[height, width, 1, features])
                            b = tf.get_variable(name='b', shape=[features])
                            conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
                                char_embeddings_conv, W, strides=[1, 1, 1, 1], padding='VALID', name='conv'), b))
                            pool = tf.reduce_max(conv, axis=1, keep_dims=True)
                            pools.append(pool)
                    pool = tf.concat(pools, -1)
                    pool = tf.reshape(pool, [batch_size, sentence_length, sum_features])
                    drop = tf.nn.dropout(pool, self.dropout)

                def dense(input, output_size, variable_scope=""):
                    with tf.variable_scope(variable_scope):
                        W = tf.get_variable("W", [input.get_shape()[1], output_size], dtype=input.dtype)
                        b = tf.get_variable("b", [output_size], dtype=input.dtype)
                    return tf.matmul(input, W) + b

                with tf.variable_scope('char-highway'):
                    _input = tf.reshape(drop, [batch_size * sentence_length, sum_features])
                    num_layers = self.config.highway_layers
                    carry_bias = -1
                    for layer in range(num_layers):
                        with tf.variable_scope('char-highway-{}'.format(layer)):
                            t = tf.sigmoid(dense(_input, _input.get_shape()[-1], 'transform-gate-{}'.format(layer)) + carry_bias)
                            g = tf.nn.relu(dense(_input, _input.get_shape()[-1], 'activation-{}'.format(layer)))
                        _input = t * g + (1 - t) * _input
                    char_highway_output = tf.reshape(_input, [batch_size, sentence_length, sum_features])

                word_embeddings = tf.concat([word_embeddings, char_rnn_output, char_highway_output], axis=-1)

        """
        Defines self.logits
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm),
                cell_bw=tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm),
                inputs=word_embeddings,
                sequence_length=self.sequence_lengths,
                dtype=tf.float32)
            output = tf.nn.dropout(x=tf.concat([output_fw, output_bw], axis=-1), keep_prob=self.dropout)

        with tf.variable_scope("proj"):
            pred = tf.matmul(a=tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm]),
                             b=tf.get_variable("W", dtype=tf.float32, shape=[2 * self.config.hidden_size_lstm,
                                                                             len(self.config.vocab_tags)])) \
                   + tf.get_variable("b", shape=[len(self.config.vocab_tags)], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
            nsteps = tf.shape(output)[1]
            self.logits = tf.reshape(pred, [-1, nsteps, len(self.config.vocab_tags)])

        """
        Defines self.labels_pred
        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)

        """Defines the loss"""
        if self.config.use_crf:
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(tf.boolean_mask(tensor=losses, mask=tf.sequence_mask(self.sequence_lengths)))

        # for tensorboard
        tf.summary.scalar("loss", self.loss)

        """Defines self.train_op that performs an update on a batch"""
        with tf.variable_scope("train_step"):
            _lr_m = self.config.lr_method.lower()  # lower to make sure
            if _lr_m == 'adam':  # sgd method
                optimizer = tf.train.AdamOptimizer(self.lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if self.config.clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(self.loss)

        """Defines self.sess and initialize the variables"""
        self.logger.info("Initializing tf session")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def train(self, train: CoNLLDataset, dev: CoNLLDataset) -> float:
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)
            dev: dataset

        """
        best_score = 0
        nepoch_no_imprv = 0  # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1, self.config.nepochs))

            score = self.run_epoch(train, dev, epoch)
            self.config.lr *= self.config.lr_decay  # decay learning rate

            # early stopping and saving best parameters
            if score >= best_score:
                nepoch_no_imprv = 0
                best_score = score
                self.save_session()
                self.logger.info("- new best score!")
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= self.config.nepoch_no_imprv:
                    self.logger.info("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                    break

        return best_score

    def run_epoch(self, train: CoNLLDataset, dev: CoNLLDataset, epoch: int) -> int:
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(train.get_minibatches(batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)

            _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def evaluate(self, test: CoNLLDataset) -> None:
        """Evaluate model on test set

        Args:
            test: instance of class Dataset

        """
        self.logger.info("Testing model over test set")
        metrics = self.run_evaluate(test)
        self.logger.info(msg=" - ".join(["{} {:04.2f}".format(k, v) for k, v in metrics.items()]))

    def run_evaluate(self, test: CoNLLDataset) -> Dict[str, int]:
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in test.get_minibatches(self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]
                accs += [a == b for (a, b) in zip(lab, lab_pred)]

                lab_chunks = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100 * acc, "f1": 100 * f1}

    def predict(self, words_raw: List[str]) -> List[str]:
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        idx_to_tag = {idx: tag for tag, idx in self.config.vocab_tags.items()}
        pred_ids, _ = self.predict_batch([words])
        preds = [idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds

    def predict_batch(self, sentences: List[List[int]]) -> Tuple[List[List[int]], int]:
        """
        Args:
            sentences: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(sentences, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]  # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths

    def get_feed_dict(self,
                      sentences: List[List[int]],
                      labels: List[List[int]] = None,
                      lr: float = None, dropout: float = None) -> Tuple[Dict, int]:
        """Given some data, pad it and build a feed dictionary

        Args:
            sentences: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            char_ids, word_ids = zip(*sentences)
            char_ids, word_lengths = pad_chars(char_ids)
            word_ids, sequence_lengths = pad_words(word_ids)
        else:
            word_ids, sequence_lengths = pad_words(sentences)

        # build feed dictionary
        feed = {}
        feed[self.word_ids] = word_ids
        feed[self.sequence_lengths] = sequence_lengths

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_words(labels)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)

    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output, self.sess.graph)


def shapes(tensor):
    shape = tf.shape(tensor)
    return [shape[i] for i in range(shape.get_shape()[0])]

