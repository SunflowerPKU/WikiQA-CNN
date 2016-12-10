#encoding=utf-8
import numpy as np
import scipy.sparse as sp
from collections import defaultdict, OrderedDict
import sys, re, cPickle, random, logging, argparse
from sklearn import linear_model

from process_data import WordVecs
from qa_cnn import make_cnn_data, train_qacnn

logger = logging.getLogger("qa.sent.score")

def tuning_cnn(revs, wordvecs, max_l=40, dev_refname=None, test_refname=None, dev_ofname=None, test_ofname=None):
    """
    training CNN representations for questions and answer sentences
    """
    # tuning parameters
    n_epoch = 5
    n_feature_maps = 50
    filter_hs = [2]
    filter_h = max(filter_hs)
    lam = 0.0
    datasets = make_cnn_data(revs, wordvecs.word_idx_map, max_l=max_l, filter_h=filter_h)
    
    #模型预测 生成[y预测]
    train_preds, dev_preds, test_preds = train_qacnn(datasets, U=wordvecs.W, filter_hs=filter_hs, hidden_units=[n_feature_maps,2], shuffle_batch=False, n_epochs=n_epoch, lam=lam, batch_size=20, lr_decay = 0.95, sqr_norm_lim=9)

    if dev_refname is not None and dev_ofname is not None:
        create_pred(dev_preds, dev_refname, dev_ofname)
    if test_refname is not None and test_ofname is not None:
        create_pred(test_preds, test_refname, test_ofname)
    return

def create_pred(preds, alfname, ofname, qcol=0, acol=2):
    """
    create prediction file for trec eval
    alfname: input file for alignment, containing quetion id column and answer id column
    """
    pscrs = []
    f = open(alfname, "rb")
    allines = f.readlines()
    f.close()

    of = open(ofname, "w")
    preqid = " "
    rankold = []
    result = []
    for i,(alline, pscr) in enumerate(zip(allines, preds)):
        parts = alline.strip().split()
        qid, aid = parts[qcol], parts[acol]
        if(i == len(preds)-1):
           result.append([qid, aid, pscr])
           rankold.append(pscr)
           ranknew = sorted(rankold, reverse = True)
           for item in result:
               item.append(ranknew.index(item[2]) + 1)
               of.write("%s\t%s\t%s\n"%(item[0], item[1], item[3]))

        if qid != preqid and len(rankold) != 0:
            ranknew = sorted(rankold, reverse = True)
            for item in result:
                item.append(ranknew.index(item[2]) + 1)
                of.write("%s\t%s\t%s\n"%(item[0], item[1], item[3]))
            rankold = []
            result = []
               
        result.append([qid, aid, pscr])
        rankold.append(pscr)
        preqid = qid

    of.close()


if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info('begin logging')

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="pkl file for dataset")
    parser.add_argument("--dev_refname", help="refname for dev set")
    parser.add_argument("--test_refname", help="refname for test set")
    parser.add_argument("--dev_ofname", help="output prediction for dev set")
    parser.add_argument("--test_ofname", help="output prediction for test set")
    args = parser.parse_args()
    
    print "loading data...",
    x = cPickle.load(open(args.dataset,"rb"))
    revs, wordvecs, max_l = x[0], x[1], x[2]
    max_l = 40
    print "data loaded!"

    tuning_cnn(revs, wordvecs, max_l, args.dev_refname, args.test_refname, args.dev_ofname, args.test_ofname)

    logger.info('end logging')
