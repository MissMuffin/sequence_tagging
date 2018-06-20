#!/usr/bin/env bash

to_conll() {
    cat $1 | sed '/^#/ d' | cut -f2-3 | expand -t 1 > $2
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

to_conll $DIR/NER-de-dev.tsv    $DIR/NER-de-dev-CoNLL2003.txt
to_conll $DIR/NER-de-test.tsv   $DIR/NER-de-test-CoNLL2003.txt
to_conll $DIR/NER-de-train.tsv  $DIR/NER-de-train-CoNLL2003.txt