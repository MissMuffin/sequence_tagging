from model.config import Config
from model.ner_model import NERModel


def main(char_dim=100, glove_dim=300, lm_dim=50, glove_pretrained=True,
         lm_pretrained=True, glove_trainable=False, lm_trainable=False, run_number=1):

    # create instance of config
    config = Config(char_dim=char_dim, glove_dim=glove_dim, lm_dim=lm_dim, glove_pretrained=glove_pretrained,
                    lm_pretrained=lm_pretrained, glove_trainable=glove_trainable, lm_trainable=lm_trainable, run_number=run_number)

    # build model
    model = NERModel(config)
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    model.train(train=config.dataset_train, dev=config.dataset_dev)


if __name__ == "__main__":
    main()
