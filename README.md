# Named Entity Recognition with Tensorflow

This repo implements a NER model using Tensorflow (LSTM + CRF + chars embeddings).

State-of-the-art performance (F1 score between 90 and 91).

Check the [blog post](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)

## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```


## Model

Similar to [Lample et al.](https://arxiv.org/abs/1603.01360) and [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF



## Getting started


1. Download the fastText vectors with

```
make fasttext
```

2. Format the GermEval dataset with

```
make germeval
```

3. Build the training data, train and evaluate the model with
```
make run
```


## Details


Here is the breakdown of the commands executed in `make run`:

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/ner_model.py`

Training time on NVidia Tesla K80 is 110 seconds per epoch on CoNLL train set using characters embeddings and CRF.



## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.

