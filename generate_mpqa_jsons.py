import codecs
import logging
from data_helpers.process_mpqa2_new import get_annotations
import os
import json
from operator import add
from stanford_corenlp_pywrapper import CoreNLP
import os.path
import random

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


def preprocess():
    '''
    filename = "../corpora/database.mpqa.2.0/datasplit/filelist_train0"
    docs_train_fold_1 = codecs.open(filename, "r", encoding="utf-8").readlines()

    filename = "../corpora/database.mpqa.2.0/datasplit/filelist_test0"
    docs_test_fold_1 = codecs.open(filename, "r", encoding="utf-8").readlines()

    filename = "../corpora/database.mpqa.2.0/datasplit/doclist.mpqaOriginalSubset"
    documents = codecs.open(filename, "r", encoding="utf-8").readlines()
    eval_docs = docs_train_fold_1 + docs_test_fold_1
    dev_docs = [doc for doc in documents if doc not in eval_docs]

    all_docs = eval_docs + dev_docs
    '''

    all_docs = open('database.mpqa.2.0/doclist.mpqaOriginalSubset', 'r').readlines() + \
               open('database.mpqa.2.0/doclist.ula-luSubset', 'r').readlines() + \
               open('database.mpqa.2.0/doclist.ulaSubset', 'r').readlines() + \
               open('database.mpqa.2.0/doclist.xbankSubset', 'r').readlines()

    for doc in all_docs:
        split_and_tokenize(doc)


def split_and_tokenize(doc):
    '''
    Reads a text document, splits sentences and tokenize them with the python wrapper of the Stanford CoreNLP.
    More info: https://github.com/brendano/stanford_corenlp_pywrapper
    :param doc: path to the
    :return:
    '''
    parse_mode = "ssplit"  # tokenization and sentence splitting
    coreNlpPath = "/Users/ana/workspace/stanford_corenlp_pywrapper/stanford-corenlp-full-2017-06-09/*"

    parser = CoreNLP(parse_mode, corenlp_jars=[coreNlpPath])

    json_name = "database.mpqa.2.0/docs/" + doc.split("\n")[0] + ".json"
    if not os.path.exists(json_name):
        doc_path = "database.mpqa.2.0/docs/" + doc.split("\n")[0]
        document = codecs.open(doc_path, "r", encoding="utf-8").read()
        data_source_parse = parser.parse_doc(document)

        with open(json_name, 'w') as fp:
            json.dump(data_source_parse, fp, sort_keys=True, indent=2)


def make_mpqa_jsons(mode, fold=0, split='prior'):
    sub_path = "database.mpqa.2.0/man_anns/"
    doc_sub_path = "database.mpqa.2.0/docs/"

    logging.info("Making a %s json for the fold %s ..." % (mode, fold))

    if mode in ['train', 'test']:
        filename = "datasplit/" + split + "/filelist_" + mode + str(fold)
        docs = codecs.open(filename, "r", encoding="utf-8").readlines()

    elif mode == 'dev':
        if split == 'prior':
            filename = "datasplit/" + split + "/filelist_train0"
            docs_train_fold_1 = codecs.open(filename, "r", encoding="utf-8").readlines()

            filename = "datasplit/" + split + "/filelist_test0"
            docs_test_fold_1 = codecs.open(filename, "r", encoding="utf-8").readlines()

            filename = "datasplit/" + split + "/doclist.mpqaOriginalSubset"
            documents = codecs.open(filename, "r", encoding="utf-8").readlines()
            eval_docs = docs_train_fold_1 + docs_test_fold_1
            docs = [doc for doc in documents if doc not in eval_docs]

        if split == 'new':
            filename = "datasplit/" + split + "/filelist_" + mode
            docs = codecs.open(filename, "r", encoding="utf-8").readlines()
    else:
        mode = 'all'
        docs = open('database.mpqa.2.0/doclist.mpqaOriginalSubset', 'r').readlines() + \
               open('database.mpqa.2.0/doclist.ula-luSubset', 'r').readlines() + \
               open('database.mpqa.2.0/doclist.ulaSubset', 'r').readlines() +\
               open('database.mpqa.2.0/doclist.xbankSubset', 'r').readlines()

    data_dict = {'documents_num': len(docs)}
    stats_corpus = [0]*14

    for i, line in enumerate(docs):
        lre = sub_path + line.split("\n")[0] + "/gateman.mpqa.lre.2.0"
        sent = sub_path + line.split("\n")[0] + "/gatesentences.mpqa.2.0"
        doc = doc_sub_path + line.split("\n")[0]

        doc_corenlp_json = doc + ".json"
        with open(doc_corenlp_json) as data_file:
            doc_corenlp = json.load(data_file)
        doc_corenlp_tokenization = doc_corenlp['sentences']

        argv = [lre, doc, sent, doc_corenlp_tokenization]
        doc_dict, stats_doc = get_annotations(argv)
        stats_corpus = map(add, stats_corpus, stats_doc)
        doc_name = 'document'+str(i)
        data_dict.update({doc_name: doc_dict})

    print("# direct subjectives (DSs) with lenght smaller or equal to one character: %d" % stats_corpus[2])
    print("# DSs for which the corresponding tokenized sentence was not retrieved: %d" % stats_corpus[5])
    print("# DSs for which two corresponding tokenized sentence were retrieved: %d" % stats_corpus[4])
    print("# DSs longer than one character and with one corresponding sentence: %d" % stats_corpus[3])
    print("# holders that are not in the same sentence as the corresponding DS: %d" % stats_corpus[6])
    print("# targets that are not in the same sentence as the corresponding DS: %d" % stats_corpus[7])
    print("# holders that are not retrieved from the given annotation: %d" % stats_corpus[9])
    print("# targets that are not retrieved from the given annotation: %d" % stats_corpus[8])
    print("# number of implicit DSs: %d" % stats_corpus[0])

    implicit_one_char =  stats_corpus[12]/float(stats_corpus[0]) if stats_corpus[0] > 0 else 0
    print("percentage of implicit DSs that are one character long: %d" % implicit_one_char)

    if mode in ['train', 'test']:
        json_name = "jsons/" + split + '/' + mode + "_fold_" + str(fold) + ".json"
    else:
        json_name = "jsons/" + split + "/dev.json"

    with open(json_name, 'w') as fp:
        json.dump(data_dict, fp, sort_keys=True, indent=2)


def generate_new_split(seed=24, dev_size=100, k=4.0):
    all_docs = open('datasplit/prior//doclist.mpqaOriginalSubset', 'r').readlines()
    random.seed(seed)
    all_docs = random.sample(all_docs, len(all_docs))

    dev_docs = all_docs[:dev_size]
    eval_docs = all_docs[dev_size:]
    test_size = int(len(eval_docs) * (1/float(k)))

    dev_file = open('datasplit/new/filelist_dev', 'w')
    for doc in dev_docs:
        dev_file.write(doc)
    dev_file.close()

    for f in range(int(k)):
        test_docs = eval_docs[f*test_size:f*test_size+test_size]
        train_docs = [doc for doc in eval_docs if doc not in test_docs]

        train_file = open('datasplit/new/filelist_train'+str(f), 'w')
        for doc in train_docs:
            train_file.write(doc)
        train_file.close()

        test_file = open('datasplit/new/filelist_test'+str(f), 'w')
        for doc in test_docs:
            test_file.write(doc)
        test_file.close()


def main(nfolds, split):
    for mode in ["train", "test"]:
        for fold in range(nfolds):
            make_mpqa_jsons(mode, fold, split)

    make_mpqa_jsons("dev", range(nfolds)[0], split)


if __name__ == "__main__":
    # Logging Information
    # ==================================================
    fname = 'logs/generate_mpqa_jsons.log'
    logging.basicConfig(
        filename=fname,
        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    #generate_new_split()
    #preprocess()
    main(4, 'new')
    main(10, 'prior')
