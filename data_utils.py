from gensim.models.keyedvectors import KeyedVectors
from collections import OrderedDict
from collections import defaultdict
from collections import Counter
from copy import deepcopy
import numpy as np
import itertools
import logging
import random
import json
import os

PAD = "<PAD>"
UNK = "<UNK>"

import sys
#reload(sys)
#sys.setdefaultencoding('utf8')


def word2vec_emb_vocab(vocabulary, dim, emb_type, model='mtl'):
    global UNK
    global PAD

    if emb_type == "w2v":
        logging.info("Loading pre-trained w2v binary file...")
        w2v_model = KeyedVectors.load_word2vec_format('../embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    else:
        # convert glove vecs into w2v format: https://github.com/manasRK/glove-gensim/blob/master/glove-gensim.py
        if model == 'mtl':
            glove_file = "../embeddings/glove/glove_" + str(dim) + "_w2vformat.txt"
        if model == 'stl':
            glove_file = "embeddings/glove/glove_" + str(dim) + "_w2vformat.txt"

        w2v_model = KeyedVectors.load_word2vec_format(glove_file, binary=False)  # GloVe Model

    w2v_vectors = w2v_model.syn0

    logging.info("building embeddings for this dataset...")
    vocab_size = len(vocabulary)
    embeddings = np.zeros((vocab_size, dim), dtype=np.float32)

    embeddings[vocabulary[PAD], :] = np.zeros((1, dim))
    embeddings[vocabulary[UNK], :] = np.mean(w2v_vectors, axis=0).reshape((1, dim))

    emb_oov_count = 0
    embv_count = 0
    for word in vocabulary:
        try:
            embv_count += 1
            embeddings[vocabulary[word], :] = w2v_model[word].reshape((1, dim))
        except KeyError:
            emb_oov_count += 1
            embeddings[vocabulary[word], :] = embeddings[vocabulary[UNK], :]

    oov_prec = emb_oov_count / float(embv_count) * 100
    logging.info("perc. of vocab words w/o a pre-trained embedding: %s" % oov_prec)

    del w2v_model

    assert len(vocabulary) == embeddings.shape[0]

    return embeddings, vocabulary, oov_prec


def get_emb_vocab(sentences, emb_type, dim, word_freq, model='mtl'):
    logging.info("Building vocabulary...")
    word_counts = dict(Counter(itertools.chain(*sentences)).most_common())
    word_counts_prune = {k: v for k, v in word_counts.iteritems() if v >= word_freq}
    word_counts_list = zip(word_counts_prune.keys(), word_counts_prune.values())

    vocabulary_inv = [x[0] for x in word_counts_list]
    vocabulary_inv.append(PAD)
    vocabulary_inv.append(UNK)

    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    if emb_type == "w2v":
        emb, vocab, oov_perc = word2vec_emb_vocab(vocabulary, dim, emb_type, model)

    if emb_type == "glove":
        emb, vocab, oov_perc = word2vec_emb_vocab(vocabulary, dim, emb_type, model)
    return emb, vocab, oov_perc


def load_conll(path, threshold):
    assert path

    corpus = []
    with open(path) as f:
        sent = []
        for line in f:
            es = line.rstrip().split()
            if len(es) > 1:
                word = es[0].lower()
                tag = es[1]
                syn = es[2]
                ne = es[3]
                prd = es[4]

                prop = []
                if len(es) > threshold:
                    prop = es[threshold:]

                sent.append((word, tag, syn, ne, prd, prop))
            else:
                corpus.append(sent)
                sent = []
        if sent:
            corpus.append(sent)
    return corpus


def get_word_ids(sent, vocabulary, task):
    global UNK
    word_ids = []
    for _, w in enumerate(sent):
        if task == 'srl':
            w_id = vocabulary[w[0].lower()] if w[0].lower() in vocabulary else vocabulary[UNK]
        else:
            w_id = vocabulary[w.lower()] if w.lower() in vocabulary else vocabulary[UNK]
        word_ids.append(w_id)
    return word_ids


def transform_srl_data(corpus, window_size, vocabulary, exp_setup_id, fold, a_dict=None, flag='flexi', mode=None):
    global UNK
    global PAD
    if a_dict:
        label_dict = deepcopy(a_dict)
    else:
        label_dict = {}

    predicates_all = []
    ctx_all = []
    ctx_len = []
    sentences = []
    labels = []
    m_all = []

    eval_orl_sent = eval_orl_sentences(exp_setup_id, fold)
    for i, sent in enumerate(corpus):
        sent_transform = [w[0].lower() for w in sent]
        if mode and sent_transform in eval_orl_sent:
            continue

        predicates = [(k, w[0]) for k, w in enumerate(sent) if w[5][0] != '-']

        if len(predicates) > 0:
            for j, (p_id, p_w) in enumerate(predicates):
                x = get_word_ids(sent, vocabulary, 'srl')
                sentences.append(x)
                y, label_dict = get_srl_labels(j, sent, label_dict, flag)
                labels.append(y)

                try:
                    p_vocab_id = vocabulary[p_w]
                except KeyError:
                    p_vocab_id = vocabulary[UNK]

                predicates_all.append([p_vocab_id])

                assert p_id < len(sent)
                assert len(x) == len(y)

                ctx = []
                end = min(len(sent), p_id+window_size+1)
                begin = max(0, p_id-window_size)
                for k in range(begin, end):
                    w = sent[k][0]
                    try:
                        w_id = vocabulary[w]
                    except KeyError:
                        w_id = vocabulary[UNK]
                    ctx.append(w_id)
                ctx_len.append(len(ctx))

                if len(sent) < p_id+window_size+1:
                    diff = p_id+window_size+1-len(sent)
                    for _ in range(diff):
                        ctx.append(vocabulary[PAD])

                if p_id-window_size < 0:
                    diff = -(p_id-window_size)
                    for _ in range(diff):
                        ctx.insert(0, vocabulary[PAD])

                m = []
                for j in range(len(sent)):
                    if j in range(p_id - window_size, p_id + window_size + 1):
                        m.append(1)
                    else:
                        m.append(0)
                m_all.append(m)

                ctx_all.append(ctx)

    assert len(sentences) == len(labels) == len(predicates_all) == len(ctx_all) == len(ctx_len) == len(m_all)

    return zip(sentences, labels, predicates_all, ctx_all, ctx_len, m_all), label_dict


def get_srl_labels(prd_i, sent, arg_dict, flag):
    sent_args = []
    prev = None
    for (j, w) in enumerate(sent):
        arg = w[5][prd_i+1]
        if arg.startswith('('):
            if arg.endswith(')'):
                prev = arg[1:-2]
                arg_label = 'B-' + prev

                if arg_label not in arg_dict:
                    if flag == 'flexi':
                        label_dict_size = len(arg_dict)
                        arg_dict[arg_label] = label_dict_size
                    if flag == 'fix':
                        arg_label = 'O'

                sent_args.append(arg_dict[arg_label])
                prev = None
            else:
                prev = arg[1:-1]
                arg_label = 'B-' + prev

                if arg_label not in arg_dict:
                    if flag == 'flexi':
                        label_dict_size = len(arg_dict)
                        arg_dict[arg_label] = label_dict_size
                    if flag == 'fix':
                        arg_label = 'O'

                sent_args.append(arg_dict[arg_label])
        else:
            if prev:
                arg_label = 'I-' + prev

                if arg_label not in arg_dict:
                    if flag == 'flexi':
                        label_dict_size = len(arg_dict)
                        arg_dict[arg_label] = label_dict_size
                    if flag == 'fix':
                        arg_label = 'O'

                sent_args.append(arg_dict[arg_label])
                if arg.endswith(')'):
                    prev = None
            else:
                arg_label = 'O'

                if arg_label not in arg_dict:
                    if flag == 'flexi':
                        label_dict_size = len(arg_dict)
                        arg_dict[arg_label] = label_dict_size
                    if flag == 'fix':
                        arg_label = 'O'
                sent_args.append(arg_dict[arg_label])
    return sent_args, arg_dict


def transform_orl_data(corpus, vocabulary, window_size, mode, exp_setup_id='new', att_link_exists_obligatory='false'):
    global UNK
    global PAD

    vocabulary_inv = [None] * len(vocabulary)
    for w in vocabulary:
        vocabulary_inv[vocabulary[w]] = w

    labels_all = []
    sentences_all = []
    ctx_all = []
    ctx_len = []
    ds_all = []
    ds_len = []
    m_all = []
    ds_indices_all = []
    sentences_original_all = []

    # data_file = open('mpqa_files/examples_' + mode + '_' + str(fold) + '.txt', 'w')
    type_counts = defaultdict(int)

    stats_dict = defaultdict(int)
    for doc_num in range(corpus['documents_num']):
        document_name = "document" + str(doc_num)
        doc = corpus[document_name]
        for sent_num in range(doc['sentences_num']):
            sentence_name = "sentence" + str(sent_num)
            sentence = doc[sentence_name]['sentence_tokenized']

            annos = []
            for ds_id in doc[sentence_name]['dss_ids']:
                ds_name = "ds" + str(ds_id)
                ds = doc[sentence_name][ds_name]
                # ds_all_num += 1
                stats_dict['ds_all_num'] += 1

                '''
                Do not allow implicit direct-subjectives!
                Example: "But there can not be any real [talk]_opinion of success until the broad strategy agains terrorism 
                         begins to bear fruit." --> attitude covers the whole sentence (no target) and reflects 
                         the attitude of the author of the document (no explicit holder) 
                '''
                if ds['ds_implicit']:
                    # implicit_ds += 1
                    stats_dict['implicit_ds'] += 1
                    continue

                '''
                Do not allow inferred direct-subjectives! 
                The task we are tackling is labelling of opinion roles of EXPLICIT opinion expressions. 
                Opinion roles of inferred opinion expressions can not be recovered with a same method. 
                '''

                # if ds['att_num'] == 1 and ds['att0']['attitudes_inferred'] == 'yes':

                inferred = True
                for atid in range(ds['att_num']):
                    if ds['att' + str(atid)]['attitudes_inferred'] != 'yes':
                        inferred = False

                if inferred:
                    # inferred_ds += 1
                    stats_dict['inferred_ds'] += 1
                    continue

                '''
                Annotation of direct-subjective may miss the attitude-link attribute, which makes impossible to trace
                its target.

                Example: 
                2053	3765,3769	string	GATE_direct-subjective	 nested-source="w,mccoy" intensity=""

                Removing such direct-subjectives is optional. We keep them.
                '''
                if not ds['attitude_link_exists']:
                    # no_att_link += 1
                    stats_dict['no_att_link'] += 1
                    if att_link_exists_obligatory == 'true':
                        continue

                '''
                A direct-subjective can be marked with the insubstantial attribute if it is:
                (1) not significant in the discourse:
                    Example: "it completely supports the [U.S]_holder [stance]_opinion" we do not know what is U.S.'s stance
                (2) not real within the discourse:
                    Example: Antonio Martino, meanwhile, said [...] that his country would not support an attack
                             on Iraq without 'proven proof' that [Baghdad]_holder is [supporting]_opinion [al Qaeda]_target.
                    --> we do not have a proof that Baghdad is supporting al Qaeda

                We keep insubstantial DSEs and label their opinion roles.
                '''
                if ds['ds_insubstantial'] != 'none':
                    # insubs_ds += 1
                    stats_dict['insubs_ds'] += 1

                '''
                A direct-subjective can have attribute 'ds_subjective_uncertain' and 'annotation-uncertain'.
                We did not discard those believing that they would have been discarded by the corpus creators if 
                they are really incorrect.
                Same holds for targets and holders. 
                '''
                if ds['ds_annotation_uncertain'] == 'somewhat-uncertain':
                    # ds_uncertain[0] += 1
                    stats_dict['ds_annotation_somewhat_uncertain'] += 1

                if ds['ds_annotation_uncertain'] == 'very-uncertain':
                    # ds_uncertain[1] += 1
                    stats_dict['ds_annotation_very_uncertain'] += 1

                ds_entity = ds['ds_tokenized']
                ds_indices = ds['ds_indices']
                assert ds_indices

                if len(set(ds_indices) & set(annos)) > 0:
                    # overlap_count += 1
                    stats_dict['overlap_count'] += 1
                annos.extend(ds_indices)

                '''
                We do not allow a holder to overlap with the corresponding direct-subjective. 
                Example: Mugabe said [Zimbabwe]_target needed their continued support against what he called [hostile
                         [international]_holder attention]_opinion.
                '''

                # do not allow overlapping holders (can mess up the BIO scheme)
                holder_indices = []
                holder_unique_ids = []
                for i, (hol1, o) in enumerate(zip(ds['holders_indices'], ds['holder_ds_overlap'])):
                    if not o:
                        not_overlap = True
                        for hol2 in holder_indices:
                            if len(set(hol1) & set(hol2)) > 0:
                                not_overlap = False

                        if not_overlap:
                            holder_indices.append(hol1)
                            holder_unique_ids.append(i)

                for hid in holder_indices:
                    if len(set(hid) & set(annos)) > 0:
                        # overlap_count += 1
                        stats_dict['overlap_count'] += 1
                    annos.extend(hid)

                holders = [hol for i, hol in enumerate(ds['holders_tokenized']) if i in holder_unique_ids]

                for u in ds['holders_uncertain']:
                    if u == 'somewhat-uncertain':
                        # holder_uncertain[0] += 1
                        stats_dict['holder_somewhat_uncertain'] += 1
                    if u == 'vert-uncertain':
                        # holder_uncertain[1] += 1
                        stats_dict['holder_very_uncertain'] += 1

                # no duplicate holders allowed
                assert len([' '.join(hol) for hol in holders]) == len(list(set([' '.join(hol) for hol in holders])))

                targets = []
                target_indices = []
                attitudes = []
                ds_attitudes = {}

                if ds['attitude_link_exists']:
                    for aid in range(ds['att_num']):
                        if ds['att' + str(aid)]['attitudes_inferred'] != 'yes':
                            # we associate only one attitude per attitude type
                            ds_attitudes[ds['att' + str(aid)]['attitudes_types']] = aid

                    '''
                    A direct-subjective can have multiple attitudes and each attitude can point to different targets. 
                    We have to pick one attitude and non-overlapping targets.
                    We chose attitudes according to the following priorities: sentiment, intention, agreement,
                                                                              arguing, other-attitude, speculation.
                    '''

                    if ds_attitudes.keys():
                        atypes = ['sentiment', 'intention', 'agree', 'arguing', 'other-attitude', 'speculation']
                        att_not_found = True
                        att_idx = []
                        for tid, atype in enumerate(atypes):
                            if att_not_found:
                                if tid < 4:  # sentiment-pos, sentiment-neg, intention-pos, etc.
                                    for polarity in ['pos', 'neg']:
                                        atype_full = atype + '-' + polarity
                                        if atype_full in ds_attitudes:
                                            att_idx.append(ds_attitudes[atype_full])
                                            attitudes.append(atype_full)
                                            att_not_found = False
                                            type_counts[atype_full] += 1
                                else:
                                    if atype in ds_attitudes:
                                        att_idx.append(ds_attitudes[atype])
                                        attitudes.append(atype)
                                        att_not_found = False
                                        type_counts[atype] += 1
                        for aid in att_idx:
                            att = ds['att' + str(aid)]

                            # do not allow overlapping targets (can mess up the BIO scheme)
                            targets_ind_temp = []
                            target_unique_ids = []
                            for i, (tar1, o) in enumerate(zip(att['targets_indices'], att['target_ds_overlap'])):
                                if not o:
                                    not_overlap = True
                                    for tar2 in targets_ind_temp:
                                        if len(set(tar1) & set(tar2)) > 0:
                                            not_overlap = False

                                    if not_overlap:
                                        targets_ind_temp.append(tar1)
                                        target_unique_ids.append(i)
                            target_indices.extend(targets_ind_temp)

                            targets_temp = [tar for i, tar in enumerate(att['targets_tokenized']) if
                                            i in target_unique_ids]
                            targets.extend(targets_temp)

                            for u in att['targets_uncertain']:
                                if u == 'somewhat-uncertain':
                                    # target_uncertain[0] += 1
                                    stats_dict['target_somewhat_uncertain'] += 1
                                if u == 'vert-uncertain':
                                    # target_uncertain[1] += 1
                                    stats_dict['target_very_uncertain'] += 1

                        for tid in target_indices:
                            if len(set(tid) & set(annos)) > 0:
                                # overlap_count += 1
                                stats_dict['overlap_count'] += 1
                            annos.extend(tid)

                '''
                Start BIO annotations:
                O = 0
                B_DS = 1
                I_DS = 2
                B_H = 3
                I_H = 4
                B_T = 5
                I_T = 6
                '''
                labels = [0] * len(sentence)
                labels[ds_indices[0]] = 1
                if len(ds_indices) > 1:
                    for idx in ds_indices[1:]:
                        labels[idx] = 2

                for holder in holder_indices:
                    labels[holder[0]] = 3
                    if len(holder) > 1:
                        for idx in holder[1:]:
                            labels[idx] = 4

                for target in target_indices:
                    labels[target[0]] = 5
                    if len(target) > 1:
                        for idx in target[1:]:
                            labels[idx] = 6

                ds_vocab_ids = []
                for w in ds_entity:
                    w_id = vocabulary[w.lower()] if w in vocabulary else vocabulary[UNK]
                    ds_vocab_ids.append(w_id)

                # context of the direct-subjective (input to the model)
                ctx = []
                end = min(len(sentence), ds_indices[len(ds_indices) - 1] + window_size + 1)
                begin = max(0, ds_indices[0] - window_size)

                for k in range(begin, end):
                    w = sentence[k]
                    w_id = vocabulary[w.lower()] if w in vocabulary else vocabulary[UNK]
                    ctx.append(w_id)

                if len(sentence) < ds_indices[len(ds_indices) - 1] + window_size + 1:
                    diff = ds_indices[len(ds_indices) - 1] + window_size + 1 - len(sentence)
                    for _ in range(diff):
                        ctx.append(vocabulary[PAD])

                if ds_indices[0] - window_size < 0:
                    diff = -(ds_indices[0] - window_size)
                    for _ in range(diff):
                        ctx.insert(0, vocabulary[PAD])

                # indicator function: 1 if a word is in the context of the DS, 0 otherwise
                m = []
                for j in range(len(sentence)):
                    if j in range(ds_indices[0] - window_size, ds_indices[len(ds_indices) - 1] + window_size + 1):
                        m.append(1)
                    else:
                        m.append(0)

                '''
                Due to to the absence of punctuation for transcripts of spoken conversations the sentence splitter
                treats a whole document as if it were one sentence.
                Therefore, for sentences longer than 150 tokens, we take 15 tokens preceding the direct-subjective,
                the expression itself and 15 tokens after s proxy for a sentence that we present to the model.
                '''
                if len(sentence) > 150:
                    cut_num = 15
                    sentence_cut = sentence[
                                   max(0, ds_indices[0] - cut_num):min(ds_indices[-1] + cut_num, len(sentence))]
                    labels_cut = labels[max(0, ds_indices[0] - cut_num):min(ds_indices[-1] + cut_num, len(sentence))]
                    m_cut = labels[max(0, ds_indices[0] - cut_num):min(ds_indices[-1] + cut_num, len(sentence))]

                    sentences_original_all.append(' '.join(sentence_cut))
                    m_all.append(m_cut)
                    labels_all.append(labels_cut)
                    sentence_vocab_ids = get_word_ids(sentence_cut, vocabulary, 'orl')
                    sentences_all.append(sentence_vocab_ids)

                else:
                    sentences_original_all.append(' '.join(sentence))
                    sentence_vocab_ids = get_word_ids(sentence, vocabulary, 'orl')
                    m_all.append(m)
                    labels_all.append(labels)
                    sentences_all.append(sentence_vocab_ids)

                ctx_len.append(len(ctx))
                ctx_all.append(ctx)
                ds_all.append(ds_vocab_ids)
                ds_indices_all.append(ds_indices)
                ds_len.append(len(ds_vocab_ids))

                '''                
                sentence_str = ' '.join([vocabulary_inv[wid] for wid in sentence_vocab_ids])
                ds_str = ' '.join([vocabulary_inv[wid] for wid in ds_vocab_ids])
                attitudes_str = '###'.join(attitudes)
                org_attitude_str = '###'.join(ds_attitudes.keys())
                holders_str = '###'.join([' '.join(holder) for holder in holders])
                targets_str = '###'.join([' '.join(target) for target in targets])
                data_file.write('\t'.join([sentence_str, ds_str, attitudes_str, org_attitude_str, holders_str, targets_str]) + '\n')
                '''

                stats_dict['ds_num_after_filter'] += 1
                if not holders and targets:
                    stats_dict['ds_no_holder'] += 1
                if not targets and holders:
                    stats_dict['ds_no_target'] += 1
                if not holders and not targets:
                    stats_dict['ds_no_roles'] += 1
                stats_dict['num_holder_after_filter'] += len(holders)
                stats_dict['num_target_after_filter'] += len(target)

    data_path = 'mpqa_files/' + exp_setup_id + '/'
    data_path = os.path.join(os.path.dirname(__file__), data_path)

    stats_file = open(data_path + 'stats_' + mode + '.txt', 'a')

    att_str = ['sentiment-neg', 'sentiment-pos', 'arguing-pos', 'other-attitude', 'intention-pos', 'arguing-neg',
               'agree-pos', 'speculation', 'agree-neg', 'intention-neg']
    att_count = [type_counts[att] for att in att_str]

    stats_dict = OrderedDict((k, v) for k, v in sorted(stats_dict.items(), key=lambda x: x[0]))

    head_file = open(data_path + 'header.txt', 'w')
    head_file.write('\t'.join(list(stats_dict.keys())) + '\t')
    head_file.write('\t'.join(att_str) + '\n')

    stats_file.write('\t'.join([str(x) for x in stats_dict.values()]) + '\t')
    stats_file.write('\t'.join([str(c) for c in att_count]) + '\n')
    stats_file.close()
    # data_file.close()

    assert len(sentences_all) == len(labels_all) == len(ds_all) == len(ds_len) == len(ctx_all) == len(ctx_len) == len(
        m_all)
    return zip(sentences_all, labels_all, ds_all, ds_len, ctx_all, ctx_len,
               m_all), ds_indices_all, sentences_original_all


def train_data_iter(srl_data, orl_data, batch_size, vocabulary, srl_label_dict, n_epochs):
    srl_data = random.sample(list(srl_data), len(srl_data))
    orl_data = random.sample(list(orl_data), len(orl_data))

    batches = [[], []]
    for task_id, data in enumerate([srl_data, orl_data]):
        if len(data) % float(batch_size) == 0:
            num_batches = int(len(data) / batch_size)
        else:
            num_batches = int(len(data) / batch_size) + 1

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(data))
            batch = data[start_index:end_index]
            if task_id == 0:
                batch_padded = pad_srl_data(batch, vocabulary, srl_label_dict)
            else:
                batch_padded = pad_orl_data(batch, vocabulary)
            batches[task_id].append(batch_padded)

    for it in range(2 * n_epochs):
        task_id = 0 if it % 2 == 0 else 1
        randint = random.sample(range(len(batches[task_id])), 1)[0]
        batch = batches[task_id][randint]
        yield batch


def stl_orl_train_data_iter(orl_data, batch_size, vocabulary, n_epochs):
    orl_data = random.sample(list(orl_data), len(orl_data))

    if len(orl_data) % float(batch_size) == 0:
        num_batches = int(len(orl_data) / batch_size)
    else:
        num_batches = int(len(orl_data) / batch_size) + 1

    batches = []
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(orl_data))
        batch = orl_data[start_index:end_index]
        batch_padded = pad_orl_data(batch, vocabulary)
        batches.append(batch_padded)

    for it in range(n_epochs):
        randint = random.sample(range(len(batches)), 1)[0]
        batch = batches[randint]
        yield batch


def eval_data_iter(data, batch_size, vocabulary, srl_label_dict):
    if len(data) % float(batch_size) == 0:
        num_batches = int(len(data) / batch_size)
    else:
        num_batches = int(len(data) / batch_size) + 1

    batches = []
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, len(data))
        batch = data[start_index:end_index]
        if srl_label_dict:
            batch_padded = pad_srl_data(batch, vocabulary, srl_label_dict)
        else:
            batch_padded = pad_orl_data(batch, vocabulary)
        batches.append(batch_padded)
    return batches


def pad_srl_data(batch, vocab, label_dict):
    global PAD
    pad_id = vocab[PAD]

    batch_x, _, _, _, _, _ = zip(*batch)
    max_length = max([len(inst) for inst in batch_x])

    batch_pad = []
    for (x, y, p_id, ctx, ctx_len, m) in batch:
        assert len(x) == len(y)
        diff = max_length - len(x)
        assert diff >= 0

        z = []
        w = []
        p = []
        for _ in range(diff):
            z.append(pad_id)
            w.append(label_dict['<PAD>'])
            p.append(0)

        p_id_len = 1
        batch_pad.append((np.concatenate([x, z], 0), np.concatenate([y, w], 0), p_id, p_id_len, ctx, ctx_len,
                          np.concatenate([m, p], 0), len(x)))

        assert len(z) == len(w)

    return batch_pad


def pad_orl_data(batch, vocab):
    global PAD
    pad_id = vocab[PAD]
    batch_x, _, _, _, _, _, _ = zip(*batch)
    max_length = max([len(inst) for inst in batch_x])

    _, _, ds_all, _, _, _, _ = zip(*batch)
    max_length_ds = max([len(ds) for ds in ds_all])

    _, _, _, _, ctx_all, _, _ = zip(*batch)
    max_length_ctx = max([len(ctx) for ctx in ctx_all])

    batch_pad = []
    for (x, y, ds, ds_len, ctx, ctx_len, m) in batch:
        assert len(x) == len(y)
        diff = max_length - len(x)
        assert diff >= 0
        z = []
        w = []
        p = []
        for _ in range(diff):
            z.append(pad_id)
            w.append(7)
            p.append(0)

        diff_ds = max_length_ds - len(ds)
        assert diff_ds >= 0
        q = []
        for _ in range(diff_ds):
            q.append(pad_id)

        diff_ctx = max_length_ctx - len(ctx)
        assert diff_ctx >= 0
        r = []
        for _ in range(diff_ctx):
            r.append(pad_id)

        batch_pad.append((np.concatenate([x, z], 0), np.concatenate([y, w], 0), np.concatenate([ds, q], 0), ds_len,
                          np.concatenate([ctx, r], 0), ctx_len, np.concatenate([m, p], 0), len(x)))

        assert len(z) == len(w)

    return batch_pad


def eval_orl_sentences(exp_setup_id, fold):
    json_name = 'jsons/' + str(exp_setup_id) + '/test_fold_' + str(fold) + '.json'
    with open(json_name) as data_file:
        orl_test_corpus = json.load(data_file)

    test_sentences_orl = []
    for doc_num in range(orl_test_corpus['documents_num']):
        document_name = 'document' + str(doc_num)
        doc = orl_test_corpus[document_name]

        for sent_num in range(doc['sentences_num']):
            sentence_name = 'sentence' + str(sent_num)
            sentence_lower = map(lambda x: x.lower(), doc[sentence_name]['sentence_tokenized'])
            test_sentences_orl.append(sentence_lower)

    json_name = 'jsons/' + str(exp_setup_id) + '/dev.json'
    with open(json_name) as data_file:
        orl_dev_corpus = json.load(data_file)

    dev_sentences_orl = []
    for doc_num in range(orl_dev_corpus['documents_num']):
        document_name = 'document' + str(doc_num)
        doc = orl_dev_corpus[document_name]

        for sent_num in range(doc['sentences_num']):
            sentence_name = 'sentence' + str(sent_num)
            sentence_lower = map(lambda x: x.lower(), doc[sentence_name]['sentence_tokenized'])
            dev_sentences_orl.append(sentence_lower)

    eval_sentences = test_sentences_orl + dev_sentences_orl

    return eval_sentences