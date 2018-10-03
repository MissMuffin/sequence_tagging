germeval:
	sh ./data/germeval2014/germeval-to-conll.sh

glove:
	wget -P ./data/ "http://nlp.stanford.edu/data/glove.6B.zip"
	unzip ./data/glove.6B.zip -d data/glove.6B/
	rm ./data/glove.6B.zip

glove840B:
	wget -P ./data/ "https://nlp.stanford.edu/data/glove.840B.300d.zip"
	unzip ./data/glove.840B.300d.zip -d data/glove.840B.300d/
	rm ./data/glove.840B.300d.zip

fasttext:
    wget -P ./data/ "https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec"

run:
	python build_data.py
	python train.py
	python evaluate.py
