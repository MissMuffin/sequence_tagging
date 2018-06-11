from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    def conll_dataset(filename):
        return CoNLLDataset(filename, config.processing_word, config.processing_tag, config.max_iter)

    model.train(train=conll_dataset(config.filename_train),
                dev=conll_dataset(config.filename_dev))


if __name__ == "__main__":
    main()
