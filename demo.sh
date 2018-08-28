echo "please using python 2.7"
if [ ! -e text8 ]; then
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/text8.zip
  else
    curl -O http://mattmahoney.net/dc/text8.zip
  fi
  unzip text8.zip
  rm text8.zip
fi

WindowSize=15
MinimalCount=5
CORPUS_NAME=text8
#CORPUS=~/build_corpus/en/1M_corpus
CORPUS=text8
PMIFolderName=${CORPUS_NAME}.win${WindowSize}.thr${MinimalCount}
WhichMatrix=PPMI
rhoForPMF=0.0625
lambdaForPMF=0.00048828125
rank=30
x_max=10
thread=8
iterationForPMF=15
save_each_iteration=0
glove_weight=1
glove_bias=1
output_file_name=demo
output_folder_name=output

rm -rf ./$PMIFolderName
mkdir ./$PMIFolderName

echo "corpus to pairs"
python ./hyperwords/corpus2pairs.py --win $WindowSize --thr $MinimalCount  --dyn --sub 1e-5 --del ${CORPUS} >./$PMIFolderName/pairs

echo "pairs to counts"
./hyperwords/pairs2counts.sh ./$PMIFolderName/pairs > ./$PMIFolderName/counts

echo "count to vocab"
python ./hyperwords/counts2vocab.py ./$PMIFolderName/counts

echo "count to pmi"
python ./hyperwords/counts2pmi.py --cds 0.75 ./$PMIFolderName/counts ./$PMIFolderName/pmi

mv ./$PMIFolderName/pmi.cooccurrence ./$PMIFolderName/cooccurrence_for_PMI
mv ./$PMIFolderName/pmi.PMI ./$PMIFolderName/matrix_for_PMI
mv ./$PMIFolderName/pmi.PPMI ./$PMIFolderName/matrix_for_PPMI
mv ./$PMIFolderName/pmi.PPMIcooccurrence ./$PMIFolderName/cooccurrence_for_PPMI

lineOfVocab=$(cat <./$PMIFolderName/pmi.words.vocab |wc -l)
numberOfNonZero=$(cat <./$PMIFolderName/cooccurrence_for_PMI |wc -l)

cat>./$PMIFolderName/meta_for_PMI<<EOF
$lineOfVocab $lineOfVocab
$numberOfNonZero training.ratings
$numberOfNonZero test.ratings
EOF

lineOfVocab=$(cat <./$PMIFolderName/pmi.words.vocab |wc -l)
numberOfNonZero=$(cat <./$PMIFolderName/cooccurrence_for_PPMI |wc -l)

cat>./$PMIFolderName/meta_for_PPMI<<EOF
$lineOfVocab $lineOfVocab
$numberOfNonZero training.ratings
$numberOfNonZero test.ratings
EOF

cd ./libpmf-1.6/
matrix_folder_name=matrix_folder_$output_folder_name
count_folder_name=count_folder_$output_folder_name

rm -rf $matrix_folder_name
rm -rf $count_folder_name
rm -rf $output_folder_name

mkdir $matrix_folder_name
mkdir $count_folder_name
ln -sf  ../../$PMIFolderName/meta_for_$WhichMatrix  ./$matrix_folder_name/meta
ln -sf  ../../$PMIFolderName/matrix_for_$WhichMatrix  ./$matrix_folder_name/training.ratings 
ln -sf  ../../$PMIFolderName/matrix_for_$WhichMatrix  ./$matrix_folder_name/test.ratings 

ln -sf  ../../$PMIFolderName/meta_for_$WhichMatrix  ./$count_folder_name/meta
ln -sf  ../../$PMIFolderName/cooccurrence_for_$WhichMatrix  ./$count_folder_name/training.ratings 
ln -sf  ../../$PMIFolderName/cooccurrence_for_$WhichMatrix  ./$count_folder_name/test.ratings 

make clean
make

echo "start matrix factorization"
./omp-pmf-train -s 10  -n $thread  -f 0 -t $iterationForPMF -q 1  -p 0 -r $rhoForPMF -l $lambdaForPMF -b 0 -k $rank  -E $save_each_iteration -X $x_max -W  $glove_weight -G $glove_bias $matrix_folder_name $count_folder_name $output_folder_name $output_file_name 

cd $output_folder_name 
yoav_or_glove=../../$PMIFolderName/pmi.words.vocab
python ../../hyperwords/extract_model_file_name.py no_use

echo "start evaluating"
cat model_name | while read line; 
do 
echo $line;
cp  $yoav_or_glove  ./${line}.words.vocab;
cp  $yoav_or_glove  ./${line}.contexts.vocab;
python ../../hyperwords/restore_matrix_to_npy_file.py $line;
echo 'this is analogy for google.txt'
python ../../hyperwords/analogy_eval.py --w+c embedding $line ../../hyperwords/testsets/analogy/google.txt;
echo 'this is ws for ws353.txt'
python ../../hyperwords/ws_eval.py --w+c embedding $line ../../hyperwords/testsets/ws/ws353.txt;
echo 'this is ws for ws353_similarity.txt'
python ../../hyperwords/ws_eval.py --w+c embedding $line ../../hyperwords/testsets/ws/ws353_similarity.txt;
echo 'this is ws for ws353_relatedness.txt'
python ../../hyperwords/ws_eval.py --w+c embedding $line ../../hyperwords/testsets/ws/ws353_relatedness.txt;
echo 'this is ws for radinsky_mturk.txt'
python ../../hyperwords/ws_eval.py --w+c embedding $line ../../hyperwords/testsets/ws/radinsky_mturk.txt;
echo 'this is ws for bruni_men.txt'
python ../../hyperwords/ws_eval.py --w+c embedding $line ../../hyperwords/testsets/ws/bruni_men.txt;
done

cd ../
make clean
rm -rf $matrix_folder_name
rm -rf $count_folder_name
rm -rf ../$output_folder_name 
mv  $output_folder_name ../

