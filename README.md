# WikiQA-CNN
Implementation of the Adaboost algorithm for  letter recognition, which is a multi classification problem.
Given the problem q and the set of sentences A, identify all the answers to q in A and sort all the sentences. 
Put the answer before the non-answer. The answers are ranked random, and the non-answers' ranks are also free.

Based on CNN, I implemented a QA system, and the MRR and MAP on dev_set is over 0.64.

You need to prepare the needed files and put them in the "./data/trec" folder. Please refer to the files in the "./data/wiki" folder .
