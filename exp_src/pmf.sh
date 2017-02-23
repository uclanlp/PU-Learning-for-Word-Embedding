#!/bin/bash

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

cd ../GloVe-1.2/make; cd ../exp_src
cd ../libpmf-1.6/make; cd ../exp_src

if [ ! -e ../data/text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi
mv text8 ../data/text8

CORPUS=../data/text8
VOCAB_FILE=vocab.txt
COOCCURRENCE_FILE=cooccurrence.bin
COOCCURRENCE_SHUF_FILE=cooccurrence.shuf.bin
BUILDDIR=../GloVe-1.2/build
LIBPMF=../libpmf-1.6/


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
  $BUILDDIR/cooccur -ascii -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
  if [[ $? -eq 0 ]]
  then

    #### NEED to write script to prepare data for pmf
    $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
    if [[ $? -eq 0 ]]
    then
       $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
       if [[ $? -eq 0 ]]
       then
           if [ "$1" = 'matlab' ]; then
               matlab -nodisplay -nodesktop -nojvm -nosplash < ./eval/matlab/read_and_evaluate.m 1>&2 
           elif [ "$1" = 'octave' ]; then
               octave < ./eval/octave/read_and_evaluate_octave.m 1>&2 
           else
               python2.7 ../eval/python/evaluate.py
           fi
       fi
    fi
  fi
fi


##here is the script to create embedding based on pmf output

cut -d  ' ' -f 1 vocab.txt  > words
paste -d ' ' words ../libpmf-1.6/example.model > pmfembedding.txt




