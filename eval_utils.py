import numpy as np
import tensorflow as tf
from operator import add
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path


def test_step(sess, seq_model, data, task_id):
    sentences, labels, ds, ds_len, ctx, ctx_len, m, sentence_lens = zip(*data)

    assert len(sentences) > 0
    assert len(labels) > 0

    feed_dict = {
        seq_model.sentences: list(sentences),  # batch_data_padded_x,
        seq_model.labels: list(labels),  # batch_data_padded_y,
        seq_model.sentence_lens: list(sentence_lens),  # batch_data_seqlens
        seq_model.ds: list(ds),
        seq_model.ds_len: list(ds_len),
        seq_model.ctx: list(ctx),
        seq_model.ctx_len: list(ctx_len),
        seq_model.m: list(m),
        seq_model.keep_rate_input: 1.0,
        seq_model.keep_rate_output: 1.0,
        seq_model.keep_state_rate: 1.0
    }
    transition_params_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/transition_params').outputs[0]
    unary_scores_op = tf.get_default_graph().get_operation_by_name('task'+str(task_id)+'/unary_scores').outputs[0]

    tf_unary_scores, tf_transition_params = sess.run([unary_scores_op, transition_params_op], feed_dict)

    predictions = []
    gold = []
    for i in range(len(sentences)):
        length = sentence_lens[i]
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores[i, :length, :], tf_transition_params)
        predictions.append(viterbi_sequence)
        gold.append(labels[i][:length])
    return predictions, gold


def eval_srl(batches, label_dict_inv, sess, seq_model, task_id):
    correct = 0
    p_total = 0
    r_total = 0
    for batch in batches:
        pred_train, true_train = test_step(sess, seq_model, batch, task_id)
        c_batch, p_batch, r_batch = f_measure(pred_train, true_train, label_dict_inv)
        correct += c_batch
        p_total += p_batch
        r_total += r_batch
    if p_total > 0:
        p = correct / float(p_total)
    else:
        p = 0.
    if r_total > 0:
        r = correct / float(r_total)
    else:
        r = 0.
    if p + r > 0:
        f_train = (2 * p * r) / float(p + r)
    else:
        f_train = 0.

    return p, r, f_train


def count_correct(errors):
    total = 0.0
    correct = 0
    for sent in errors:
        total += len(sent)
        for y_pred in sent:
            if y_pred == 0:
                correct += 1
    return total, correct


def get_spans(sent, label_dict_inv):
    spans = []
    span = []
    #sent = sent_array.tolist()
    for w_i, a_id in enumerate(sent):
        #label = arg_dict.get_word(int(a_id))
        label = str(label_dict_inv[int(a_id)])

        if label.startswith('B-'):
            if span and span[0][0] != 'V':
                spans.append(span)
            span = [label[2:], w_i, w_i]
        elif label.startswith('I-'):
            if span:
                if label[2:] == span[0]:
                    span[2] = w_i
                else:
                    if span[0][0] != 'V':
                        spans.append(span)
                    span = [label[2:], w_i, w_i]
            else:
                span = [label[2:], w_i, w_i]
        else:
            if span and span[0][0] != 'V':
                spans.append(span)
            span = []
    if span:
        spans.append(span)
    return spans


def count_spans(spans):
    total = 0
    for span in spans:
        if not span[0].startswith('C'):
            total += 1
    return total


def f_measure(predicts, answers, label_dict_inv):
    p_total = 0.
    r_total = 0.
    correct = 0.
    for i in range(len(predicts)):
        ys = predicts[i]
        ds = answers[i]

        y_spans = get_spans(ys, label_dict_inv)
        d_spans = get_spans(ds, label_dict_inv)

        p_total += count_spans(y_spans)
        r_total += count_spans(d_spans)

        for y_span in y_spans:
            if y_span[0].startswith('C'):
                continue
            if y_span in d_spans:
                correct += 1.

    return correct, p_total, r_total


def eval_orl(batches, sess, seq_model, task_id):
    intersect_binary = [0] * 3
    intersect_proportional = [0] * 3
    num_gold = [0] * 3
    num_pred = [0] * 3
    for batch in batches:
        pred_train, true_train = test_step(sess, seq_model, batch, task_id)
        intersect_binary_temp, intersect_proportional_temp, num_pred_temp, num_gold_temp = f_measure_orl(pred_train, true_train)
        intersect_binary = map(add, intersect_binary, intersect_binary_temp)
        intersect_proportional = map(add, intersect_proportional, intersect_proportional_temp)
        num_gold = map(add, num_gold, num_gold_temp)
        num_pred = map(add, num_pred, num_pred_temp)
    fscores_dev = f_measure_final(intersect_binary, intersect_proportional, num_pred, num_gold)
    macro_binary_fscore_dev = fscores_dev[0]
    macro_proportional_fscore_dev = fscores_dev[1]
    return macro_binary_fscore_dev, macro_proportional_fscore_dev


def f_measure_orl(prediction, target):
    beggining_label = {'d': 1, 'h': 3, 't': 5}

    intersect_binary = [0]*3
    intersect_proportional = [0]*3
    num_gold = [0]*3
    num_pred = [0]*3
    for t, type in enumerate(['d', 'h', 't']):
        gold = []
        for i in range(len(target)):
            previous = -1
            for j in range(len(target[i])):
                entity = []
                b = beggining_label[type]
                if (target[i][j] == b) or (target[i][j] == b+1 and previous not in [b, b+1]):
                    entity.append((i, j))
                    flag = 0
                    for k in range(j+1, len(target[i])):
                        if (target[i][k] == b+1) and (flag == 0):
                            entity.append((i, k))
                        else:
                            flag = 1
                    if entity:
                        gold.append(entity)
                    previous = target[i][j]

        predicted = []
        for i in range(len(prediction)):
            previous = -1
            for j in range(len(prediction[i])):
                entity = []
                b = beggining_label[type]
                if (prediction[i][j] == b) or (prediction[i][j] == b+1 and previous not in [b, b+1]):
                    entity.append((i, j))
                    flag = 0
                    for k in range(j + 1, len(prediction[i])):
                        if (prediction[i][k] == b + 1) and (flag == 0):
                            entity.append((i, k))
                        else:
                            flag = 1
                    if entity:
                        predicted.append(entity)
                    previous = prediction[i][j]

        intersect_binary_temp = 0.0
        intersect_proportional_temp = 0.0

        for entity_pred in predicted:
            flag = 0
            for entity_gold in gold:
                if (len(list(set(entity_pred) & set(entity_gold))) >= 1) and (flag == 0):
                    intersect_binary_temp += 1
                    intersect_proportional_temp += len(list(set(entity_pred) & set(entity_gold))) / float(len(entity_gold))
                    flag = 1
        intersect_binary[t] = intersect_binary_temp
        intersect_proportional[t] = intersect_proportional_temp
        num_gold[t] = len(gold)
        num_pred[t] = len(predicted)

    return intersect_binary, intersect_proportional, num_pred, num_gold


def f_measure_final(intersect_binary, intersect_proportional, num_pred, num_gold):
    precision_binary = []
    recall_binary = []
    fscore_binary = []

    precision_proportional = []
    recall_proportional = []
    fscore_proportional = []

    for t, type in enumerate(['d', 'h', 't']):
        try:
            recall_binary.append(intersect_binary[t] / float(num_gold[t]))
        except ZeroDivisionError:
            recall_binary.append(0.0)

        try:
            recall_proportional.append(intersect_proportional[t] / float(num_gold[t]))
        except ZeroDivisionError:
            recall_proportional.append(0.0)

        try:
            precision_binary.append(intersect_binary[t] / float(num_pred[t]))
        except ZeroDivisionError:
            precision_binary.append(0.0)

        try:
            precision_proportional.append(intersect_proportional[t] / float(num_pred[t]))
        except ZeroDivisionError:
            precision_proportional.append(0.0)

        try:
            fscore_binary.append(
                2.0 * precision_binary[t] * recall_binary[t] / float(precision_binary[t] + recall_binary[t]))
        except ZeroDivisionError:
            fscore_binary.append(0.0)

        try:
            fscore_proportional.append(2.0 * precision_proportional[t] * recall_proportional[t] /
                                                            float(precision_proportional[t] + recall_proportional[t]))
        except ZeroDivisionError:
            fscore_proportional.append(0.0)
    return [np.nan_to_num(fscore_binary), np.nan_to_num(fscore_proportional)]


def plot_training_curve(fig_path, num_iter, flist):
    plt.figure(dpi=400)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12
    steps = range(1, num_iter + 1)
    plt.plot(steps, flist[0], linewidth=1, color='#6699ff', linestyle='-', marker='o',
             markeredgecolor='black',
             markeredgewidth=0.5, label='train')
    plt.plot(steps, flist[1], linewidth=3, color='#ff4d4d', linestyle='-', marker='D',
             markeredgecolor='black',
             markeredgewidth=0.5, label='test')
    plt.plot(steps, flist[2], linewidth=2, color='#ffcc66', linestyle='-', marker='s',
             markeredgecolor='black',
             markeredgewidth=0.5, label='dev')
    plt.xlabel('epochs')
    plt.ylabel('binary f1')
    plt.legend(loc='best', numpoints=1, fancybox=True)
    plt.savefig(fig_path)


def record_results(fname, rlist, mode):
    action = 'w' if mode == 'dev' else 'a'
    resfile = open(fname, action)
    resfile.write('\t'.join([str(x) for x in rlist]) + '\n')
    resfile.close()

