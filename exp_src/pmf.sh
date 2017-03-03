#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

cd ../GloVe-1.2/; make; cd ../exp_src
cd ../libpmf-1.6/; make; cd ../exp_src

if [ ! -e ../data/text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi
mkdir -p ../data
mv text8 ../data/text8

CORPUS=../data/text8
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=../GloVe-1.2/build
LIBPMF=../libpmf-1.6
CONVERTER=$LIBPMF/converter
PMF=$LIBPMF/pmf-train


SAVE_FILE=vectors
VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=5
VECTOR_SIZE=50
MAX_ITER=15
WINDOW_SIZE=15
BINARY=2
NUM_THREADS=8
X_MAX=10

#### Train libpmf



$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
if [[ $? -eq 0 ]]
  then
  $BUILDDIR/cooccur -ascii 1  -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
    then
    lineOfVocab=$(cat <vocab.txt |wc -l)
    numberOfNonZero=$(cat <cooccurrence.bin |wc -l)
    cp cooccurrence.bin ../libpmf-1.6/toy-example/test.ratings
    cp cooccurrence.bin ../libpmf-1.6/toy-example/training.ratings    
cat>../libpmf-1.6/toy-example/meta<<EOF
$lineOfVocab $lineOfVocab
$numberOfNonZero training.ratings
$numberOfNonZero test.ratings
EOF
        	
    if [[ $? -eq 0 ]]
      then
      cd ../libpmf-1.6/
      ./converter toy-example
      ./omp-pmf-train -s 10 -n 12 -f 1 -q 1 -p 0 -t 10 -b 0 -k 50 toy-example
      if [[ $? -eq 0 ]]
        then
        cd ../exp_src
        cut -f 1 -d ' ' vocab.txt > word
        paste -d ' ' word  ../libpmf-1.6/toy-example.model >pmf_embedding
        if [[ $? -eq 0 ]]
          then 
          python2.7 ../eval/python/evaluate.py '--vocab_file'=vocab.txt '--vectors_file'=pmf_embedding
        fi
      fi
    fi
  fi
fi




