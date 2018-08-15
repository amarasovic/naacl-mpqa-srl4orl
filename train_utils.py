import tensorflow as tf
import numpy as np


def _train_task_ops(lr, clip, model_name):
    train_ops = []
    for task_id in range(2):
        optimizer = tf.train.AdamOptimizer(lr)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='task'+str(task_id))

        #loss = tf.get_default_graph().get_tensor_by_name('task'+str(task_id)+'/task_logl_loss:0')
        loss = tf.get_default_graph().get_tensor_by_name('task' + str(task_id) + '/total_loss:0')

        if model_name == 'asp':
            tvars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        gradients = tf.gradients(loss, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grads, _ = tf.clip_by_global_norm(gradients, clip)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        train_ops.append(train_op)
    return train_ops


def train_task_step(seq_model, sess, train_op, task_id, data, keep_rate_input, keep_rate_output,
                    keep_state_rate, model_name):
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
                 seq_model.keep_rate_input: keep_rate_input,
                 seq_model.keep_rate_output: keep_rate_output,
                 seq_model.keep_state_rate: keep_state_rate
                }

    if model_name == 'asp':
        feed_dict.update({seq_model.domain_labels: [task_id]*len(sentences)})
        total_loss = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/total_loss').outputs[0]
        transition_params_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/transition_params').outputs[0]
        unary_scores_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/unary_scores').outputs[0]

        variables_names = [v.name for v in tf.trainable_variables()]
        before = sess.run(tf.trainable_variables())
        _, tf_unary_scores, tf_transition_params = sess.run([total_loss,
                                                                #reg_cost_op,
                                                                unary_scores_op,
                                                                transition_params_op],
                                                                feed_dict)
        _ = sess.run([train_op], feed_dict)
        after = sess.run(tf.trainable_variables())

        for i, (b, a) in enumerate(zip(before, after)):
            # Make sure something changed.
            if variables_names[i].split('/')[0] in ['shared', 'discriminator']:
                assert (b != a).any()
            else:
                if int(variables_names[i].split('/')[0][-1]) == task_id:
                    assert (b != a).any()

                if int(variables_names[i].split('/')[0][-1]) == task_id:
                    assert (b != a).any()

        predictions = []
        gold = []
        for i in range(len(sentences)):
            length = sentence_lens[i]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores[i, :length, :], tf_transition_params)
            predictions.append(viterbi_sequence)
            gold.append(labels[i][:length])
        return predictions, gold, None
    else:
        task_losses_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/task_logl_loss').outputs[0]
        transition_params_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/transition_params').outputs[0]
        unary_scores_op = tf.get_default_graph().get_operation_by_name('task' + str(task_id) + '/unary_scores').outputs[0]

        variables_names = [v.name for v in tf.trainable_variables()]
        before = sess.run(tf.trainable_variables())
        _, tf_unary_scores, tf_transition_params = sess.run([task_losses_op,
                                                               #reg_cost_op,
                                                               unary_scores_op,
                                                               transition_params_op],
                                                               feed_dict)
        _ = sess.run([train_op], feed_dict)
        after = sess.run(tf.trainable_variables())
        for i, (b, a) in enumerate(zip(before, after)):
            # Make sure something changed.
            if variables_names[i].split('/')[0] == 'shared':
                assert (b != a).any()
            else:
                if int(variables_names[i].split('/')[0][-1]) == task_id:
                    assert (b != a).any()

                if int(variables_names[i].split('/')[0][-1]) == task_id:
                    assert (b != a).any()

        predictions = []
        gold = []
        for i in range(len(sentences)):
            length = sentence_lens[i]
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(tf_unary_scores[i, :length, :], tf_transition_params)
            predictions.append(viterbi_sequence)
            gold.append(labels[i][:length])
        return predictions, gold, None


