cd src

# create trec datasets and extract count features
python -u process_data.py --w2v_fname ../data/GoogleNews-vectors-negative300.bin --extract_feat 1 ../data/wiki/WikiQA-train.tsv ../data/wiki/WikiQA-dev.tsv ../data/wiki/WikiQA-test.tsv ../wiki_cnn.pkl

# ----- Convolutional Neural Networks -----
python -u qa_score.py --dev_refname ../data/wiki/WikiQA-dev.ref  --test_refname  ../data/wiki/WikiQA-test.ref --dev_ofname ../pred/wiki/cnn-dev.rank --test_ofname ../pred/wiki/cnn-test.rank ../wiki_cnn.pkl

# evulate
python eval.py ../pred/wiki/cnn-dev.rank ../data/wiki/WikiQA-dev.tsv

