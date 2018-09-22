from model.config import Config
from model.ner_model import NERModel


def main():

    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    model.train(train=config.dataset_train, dev=config.dataset_dev)


if __name__ == "__main__":
    main()
