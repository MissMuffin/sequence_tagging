import argparse
import train
import evaluate

# run.py --char_dim 100 --lm_dim 100 --glove_pretrained --lm_pretrained --runs 2

parser = argparse.ArgumentParser()

parser.add_argument("--char_dim", type=int,
                    help="Dimension for the character embedding")
# parser.add_argument("dim_glove", type=int, help="")
parser.add_argument("--lm_dim", type=int, choices=[
                    50, 100, 200, 300, 1024], help="Dimension for the langauge model word embedding")
parser.add_argument("--glove_pretrained", action="store_true",
                    help="If true, uses pretrained word embeddings from glove. Otherwise init in specified dimension as zero vector")
parser.add_argument("--lm_pretrained", action="store_true",
                    help="If true, uses pretrained word embeddings from 1 billion word langauge model. Otherwise init in specified dimension as zero vector")
parser.add_argument("--glove_trainable", action="store_true",
                    help="If false, freezes glove embeddings. Otherwise train simultaneously with NER tagging")
parser.add_argument("--lm_trainable", action="store_true",
                    help="If false, freezes language model embeddings. Otherwise train simultaneously with NER tagging")
parser.add_argument("--runs", type=int,
                    help="Number of times train and evaluate should be run")

args = parser.parse_args()

char_dim = args.char_dim
glove_dim = 300
lm_dim = args.lm_dim
glove_pretrained = args.glove_pretrained
lm_pretrained = args.lm_pretrained
glove_trainable = args.glove_trainable
lm_trainable = args.lm_trainable
number_of_runs = args.runs

for run in range(args.runs):
    train.main(char_dim=char_dim, glove_dim=glove_dim, lm_dim=lm_dim, glove_pretrained=glove_pretrained,
               lm_pretrained=lm_pretrained, glove_trainable=glove_trainable, lm_trainable=lm_trainable, run_number=run)
    evaluate.main(char_dim=char_dim, glove_dim=glove_dim, lm_dim=lm_dim, glove_pretrained=glove_pretrained,
                  lm_pretrained=lm_pretrained, glove_trainable=glove_trainable, lm_trainable=lm_trainable, run_number=run)
