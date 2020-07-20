# HFR
Here is a simple command to run the module to compute hashcode representations of sentences.

-c is for the number of cores
-f is for the path of the text file containing sentences
-g is for the path of glove 100D vectors txt file

python -m hash_sentences -c 7 -f ./hash_sentences/txt_data/sample_small.txt -g ../../glove.6B.100d.txt 
