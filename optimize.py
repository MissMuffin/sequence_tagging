from typing import List, Any, Dict

import tensorflow as tf
from model.config import Config
from model.ner_model import NERModel
from skopt import gp_minimize
from skopt.space import Integer, Categorical
from skopt.plots import plot_convergence


def main():

    dimensions = [Categorical([50, 100, 200, 300], name='dim_char'),
                  Categorical([True], name='use_pretrained_words'),
                  Categorical([True], name='use_pretrained_chars'),
                  Categorical([False], name="train_word_embeddings"),
                  Categorical([False, True], name="train_char_embeddings"),
                  Categorical([300], name='hidden_size_lstm'),
                  Integer(100, 300, name='hidden_size_char'),
                  Integer(10, 50, name='features_per_ngram'),
                  Integer(2, 6, name='max_size_ngram'),
                  Integer(1, 2, name='highway_layers')]

    def func(hyperparameters: List[Any]) -> float:
        print('hyperparameters={}'.format(hyperparameters_to_dict(hyperparameters)))
        config = Config(dim_char=hyperparameters[0])
        config.use_pretrained_words = hyperparameters[1]
        config.use_pretrained_chars = hyperparameters[2]
        config.train_word_embeddings = hyperparameters[3]
        config.train_char_embeddings = hyperparameters[4]
        config.hidden_size_lstm = hyperparameters[5]
        config.hidden_size_char = hyperparameters[6]
        config.features_per_ngram = hyperparameters[7]
        config.max_size_ngram = hyperparameters[8]
        config.highway_layers = hyperparameters[9]
        config.nepochs = 10
        return -NERModel(config).train(train=config.dataset_train, dev=config.dataset_dev)

    def hyperparameters_to_dict(hyperparameters: List[Any]) -> Dict[str, Any]:
        return {dimension.name: hyperparameter for (dimension, hyperparameter) in zip(dimensions, hyperparameters)}

    def print_results(result) -> None:
        for x_iter, func_val in zip(result.x_iters, result.func_vals):
            print('hyperparameters={} -> f1-Score={}'.format(hyperparameters_to_dict(x_iter), -func_val))

    def callback(result) -> None:
        print_results(result)
        tf.reset_default_graph()

    res = gp_minimize(func=func, dimensions=dimensions, callback=callback)
    plot_convergence(res)
    print('Best result with hyperparameters={} and f1-score={}'.format(hyperparameters_to_dict(res.x), res.fun))


if __name__ == "__main__":
    main()
