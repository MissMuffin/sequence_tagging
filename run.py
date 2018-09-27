from model.config import Config
from model.ner_model import NERModel

def do_run(runs=6, char_dim=100, glove_dim=300, lm_dim=None,
           glove_pretrained=True, lm_pretrained=True, glove_trainable=False,
           lm_trainable=False, log_suffix=None, lm_embeddings_filename=None):
    for run in range(runs):

        config = Config(char_dim=char_dim,
                        glove_dim=glove_dim,
                        lm_dim=lm_dim,
                        glove_pretrained=glove_pretrained,
                        lm_pretrained=lm_pretrained,
                        glove_trainable=glove_trainable,
                        lm_trainable=lm_trainable,
                        run_number=run+1,
                        log_suffix=log_suffix,
                        lm_embeddings_file=lm_embeddings_filename)

        model = NERModel(config)

        model.train(train=config.dataset_train, dev=config.dataset_dev)

        model.restore_session(config.dir_model)
        model.evaluate(config.dataset_test)

        model.reset_session_graph()

# BASELINE: glove d300, init chars d100, crf
do_run(log_suffix="baseline")

# baseline + train glove
do_run(glove_trainable=True, log_suffix="train_glove")

# baseline + lm d50, no train
do_run(lm_dim=50)

# baseline + lm d100, no train
do_run(lm_dim=100)

# baseline + lm d200, no train
do_run(lm_dim=200)

# baseline + lm d300, no train
do_run(lm_dim=300)

# baseline + lm d1024, no train
do_run(lm_dim=1024)

# baseline + lm d1024 (NO PCA), no train
do_run(runs=10, lm_dim=1024, log_suffix="no_pca", lm_embeddings_filename="data/lm1b_embeddings_d1024.npz")

# -> take best dimension and run:
# with training enabled
# with training enabled and no pretrained lm embeddings
# with different hyperparameters:
# hidden lstm size
# 50, 100, 300, 500
# dropout
# learning rate
