from model.conll_dataset import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        try:
            # for python 2
            sentence = raw_input("input> ")
        except NameError:
            # for python 3
            sentence = input("input> ")

        words_raw = sentence.strip().split(" ")

        if words_raw == ["exit"]:
            break

        preds = model.predict(words_raw)
        to_print = align_data({"input": words_raw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main(char_dim=100, glove_dim=300, lm_dim=50, glove_pretrained=True,
         lm_pretrained=True, glove_trainable=False, lm_trainable=False, run_number=1):
         
    # create instance of config
    config = Config(char_dim=char_dim, glove_dim=glove_dim, lm_dim=lm_dim, glove_pretrained=glove_pretrained,
                    lm_pretrained=lm_pretrained, glove_trainable=glove_trainable, lm_trainable=lm_trainable, run_number=run_number)

    # build model
    model = NERModel(config)
    model.restore_session(config.dir_model)

    # evaluate and interact
    model.evaluate(config.dataset_test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
