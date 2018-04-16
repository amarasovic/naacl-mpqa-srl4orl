import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import numpy as np
from flip_gradient import flip_gradient


class SRL4ORL_deep_tagger(object):
    def __init__(self,
                 n_classes_srl,
                 n_classes_orl,
                 embeddings,
                 embeddings_trainable,
                 hidden_size,
                 cell,
                 seed,
                 n_layers_shared,
                 n_layers_orl,
                 adv_coef,
                 reg_coef):

        # input params
        '''
        TBD
        '''

        tf.set_random_seed(seed)
        n_tasks = 2

        # input params
        self.sentences = tf.placeholder(dtype=tf.int32, shape=[None, None], name='sentences')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='labels')  # this is not onehot!
        self.sentence_lens = tf.placeholder(dtype=tf.int32, shape=[None], name='sentence_lens')  # length of every PADDED sequence in the batch
        self.ds = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ds_ids')
        self.ds_len = tf.placeholder(dtype=tf.int32, shape=[None], name='ds_len')  # length of every PADDED sequence in the batch
        self.ctx = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ctx')
        self.ctx_len = tf.placeholder(dtype=tf.int32, shape=[None], name='ctx_len')
        self.m = tf.placeholder(dtype=tf.int32, shape=[None, None], name='m')

        self.keep_rate_input = tf.placeholder(dtype=tf.float32, name='keep_rate_input')
        self.keep_rate_output = tf.placeholder(dtype=tf.float32, name='keep_rate_output')
        self.keep_state_rate = tf.placeholder(dtype=tf.float32, name='state_keep_rate')

        self.domain_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='domain_lables')

        with tf.variable_scope('embeddings_lookup'):
            flag = False
            if embeddings_trainable == 'True':
                flag = True

            if isinstance(embeddings, np.ndarray):
                self.embeddings = tf.get_variable('pretrained_emb',
                                                    shape=embeddings.shape,
                                                    initializer=tf.constant_initializer(embeddings),
                                                    trainable=flag,
                                                    dtype=tf.float32)

            embedded_tokens = tf.nn.embedding_lookup(self.embeddings, self.sentences)
            embedded_ctx = tf.nn.embedding_lookup(self.embeddings, self.ctx)
            embedded_ds = tf.nn.embedding_lookup(self.embeddings, self.ds)

            embedded_ctx_sum = tf.reduce_sum(embedded_ctx, reduction_indices=-2)
            embedded_ctx_mean = embedded_ctx_sum / tf.cast(tf.expand_dims(self.ctx_len, -1), dtype=tf.float32)
            embedded_ctx_mean_copy = tf.expand_dims(embedded_ctx_mean, 1)
            pattern = tf.stack([1, tf.shape(self.sentences)[1], 1])
            embedded_ctx_mean_copy = tf.tile(embedded_ctx_mean_copy, pattern)

            embedded_ds_sum = tf.reduce_sum(embedded_ds, reduction_indices=-2)
            embedded_ds_mean = embedded_ds_sum / tf.cast(tf.expand_dims(self.ds_len, -1), dtype=tf.float32)
            embedded_ds_mean_copy = tf.expand_dims(embedded_ds_mean, 1)
            pattern = tf.stack([1, tf.shape(self.sentences)[1], 1])
            embedded_ds_mean_copy = tf.tile(embedded_ds_mean_copy, pattern)

            inputs = tf.concat(axis=2, values=[embedded_tokens,
                                               embedded_ds_mean_copy,
                                               embedded_ctx_mean_copy,
                                               tf.cast(tf.expand_dims(self.m, 2), dtype=tf.float32)])
            inputs = tf.nn.dropout(inputs, keep_prob=self.keep_rate_input, name='inputs')

        with vs.variable_scope('shared'):
            if cell == 'gru':
                stacked_fw = []
                stacked_bw = []
                for _ in range(n_layers_shared):
                    with vs.variable_scope('forward'):
                        cell_fw = tf.contrib.rnn.GRUCell(num_units=hidden_size,
                                                         kernel_initializer=tf.orthogonal_initializer(seed=seed))
                        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                                variational_recurrent=True,
                                                                state_keep_prob=self.keep_state_rate,
                                                                output_keep_prob=self.keep_rate_output,
                                                                dtype=tf.float32)
                        stacked_fw.append(cell_fw)

                    with vs.variable_scope('backward'):
                        cell_bw = tf.contrib.rnn.GRUCell(num_units=hidden_size,
                                                         kernel_initializer=tf.orthogonal_initializer(seed=seed))
                        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                                variational_recurrent=True,
                                                                state_keep_prob=self.keep_state_rate,
                                                                output_keep_prob=self.keep_rate_output,
                                                                dtype=tf.float32)
                        stacked_bw.append(cell_bw)

            if cell == 'lstm':
                stacked_fw = []
                stacked_bw = []
                for _ in range(n_layers_shared):
                    with vs.variable_scope('forward'):
                        cell_fw = tf.contrib.rnn.LSTMCell(num_units=hidden_size,
                                                          state_is_tuple=True,
                                                          initializer=tf.orthogonal_initializer(seed=seed))
                        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                                variational_recurrent=True,
                                                                state_keep_prob=self.keep_state_rate,
                                                                output_keep_prob=self.keep_rate_output,
                                                                dtype=tf.float32)
                        stacked_fw.append(cell_fw)

                    with vs.variable_scope('backward'):
                        cell_bw = tf.contrib.rnn.LSTMCell(num_units=hidden_size,
                                                          state_is_tuple=True,
                                                          initializer=tf.orthogonal_initializer(seed=seed))
                        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                                variational_recurrent=True,
                                                                state_keep_prob=self.keep_state_rate,
                                                                output_keep_prob=self.keep_rate_output,
                                                                dtype=tf.float32)

                        stacked_bw.append(cell_bw)

            multicell_fw = tf.contrib.rnn.MultiRNNCell(stacked_fw, state_is_tuple=True)
            multicell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw, state_is_tuple=True)

            self.outputs_shared_tuple, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell_fw,
                                                                              cell_bw=multicell_bw,
                                                                              dtype=tf.float32,
                                                                              sequence_length=self.sentence_lens,
                                                                              inputs=inputs)
            outputs_fw, outputs_bw = self.outputs_shared_tuple
            self.outputs_shared = tf.concat(axis=2, values=[outputs_fw, outputs_bw])

        if adv_coef > 0.0:
            with vs.variable_scope('discriminator'):
                dscrm_weights = tf.get_variable(shape=[2 * hidden_size, 2],
                                                initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                           mode='FAN_IN',
                                                                                                           uniform=False,
                                                                                                           seed=seed,
                                                                                                           dtype=tf.float32),
                                                dtype=tf.float32,
                                                trainable=True,
                                                name='dscrm_w')

                dscrm_biases = tf.get_variable(name='dscr_b',
                                               shape=[n_tasks],
                                               initializer=tf.constant_initializer(value=0),
                                               dtype=tf.float32,
                                               trainable=True)

                outputs_shared_ = flip_gradient(self.outputs_shared)

                self.outputs_shared_aggregated = tf.reduce_sum(outputs_shared_, reduction_indices=-2)
                self.outputs_shared_mean = self.outputs_shared_aggregated / tf.cast(tf.expand_dims(self.sentence_lens, -1),
                                                                                   dtype=tf.float32)
                self.outputs_shared_reshape = tf.reshape(self.outputs_shared_mean,
                                                    shape=[tf.shape(self.sentences)[0], 2 * hidden_size])
                self.dscrm_logits = tf.matmul(self.outputs_shared_reshape, dscrm_weights) + dscrm_biases

                self.domain_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.dscrm_logits,
                                                                                  labels=self.domain_labels,
                                                                                   name='dscrm_ce_loss')

        n_classes = [n_classes_srl, n_classes_orl]
        for task_id in range(2):
            with vs.variable_scope('task' + str(task_id)):
                if cell == 'gru':
                    stacked_fw = []
                    stacked_bw = []
                    for _ in range(n_layers_orl):
                        with vs.variable_scope('forward'):
                            cell_fw = tf.contrib.rnn.GRUCell(num_units=hidden_size,
                                                             kernel_initializer=tf.orthogonal_initializer(seed=seed))
                            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                                    variational_recurrent=True,
                                                                    state_keep_prob=self.keep_state_rate,
                                                                    output_keep_prob=self.keep_rate_output,
                                                                    dtype=tf.float32)
                            stacked_fw.append(cell_fw)

                        with vs.variable_scope('backward'):
                            cell_bw = tf.contrib.rnn.GRUCell(num_units=hidden_size,
                                                             kernel_initializer=tf.orthogonal_initializer(seed=seed))
                            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                                    variational_recurrent=True,
                                                                    state_keep_prob=self.keep_state_rate,
                                                                    output_keep_prob=self.keep_rate_output,
                                                                    dtype=tf.float32)
                            stacked_bw.append(cell_bw)

                if cell == 'lstm':
                    stacked_fw = []
                    stacked_bw = []
                    for _ in range(n_layers_orl):
                        with vs.variable_scope('forward'):
                            cell_fw = tf.contrib.rnn.LSTMCell(num_units=hidden_size,
                                                              state_is_tuple=True,
                                                              initializer=tf.orthogonal_initializer(seed=seed))
                            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                                                    variational_recurrent=True,
                                                                    state_keep_prob=self.keep_state_rate,
                                                                    output_keep_prob=self.keep_rate_output,
                                                                    dtype=tf.float32)
                            stacked_fw.append(cell_fw)

                        with vs.variable_scope('backward'):
                            cell_bw = tf.contrib.rnn.LSTMCell(num_units=hidden_size,
                                                              state_is_tuple=True,
                                                              initializer=tf.orthogonal_initializer(seed=seed))
                            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                                                    variational_recurrent=True,
                                                                    state_keep_prob=self.keep_state_rate,
                                                                    output_keep_prob=self.keep_rate_output,
                                                                    dtype=tf.float32)
                            stacked_bw.append(cell_bw)

                multicell_fw = tf.contrib.rnn.MultiRNNCell(stacked_fw, state_is_tuple=True)
                multicell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw, state_is_tuple=True)

                outputs_task_tuple, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=multicell_fw,
                                                                        cell_bw=multicell_bw,
                                                                        dtype=tf.float32,
                                                                        sequence_length=self.sentence_lens,
                                                                        inputs=inputs)

                # flatten all dimension except the last; shape: (batch_size x seq_len, output_size)
                outputs_fw, outputs_bw = outputs_task_tuple
                outputs_task = tf.concat(axis=2, values=[outputs_fw, outputs_bw])
                outputs = tf.concat(values=[self.outputs_shared, outputs_task], axis=2)

                output_layer_size = 4 * hidden_size
                outputs_flat = tf.reshape(outputs, [-1, output_layer_size])

                out_weights = tf.get_variable(name='out_w',
                                               shape=[output_layer_size, n_classes[task_id]],
                                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                                                          mode='FAN_IN',
                                                                                                          uniform=False,
                                                                                                          seed=seed,
                                                                                                          dtype=tf.float32),
                                               dtype=tf.float32,
                                               trainable=True)

                out_biases = tf.get_variable(name='out_b',
                                              shape=[n_classes[task_id]],
                                              initializer=tf.constant_initializer(value=0),
                                              dtype=tf.float32,
                                              trainable=True)

                matricized_unary_scores = tf.matmul(outputs_flat, tf.nn.dropout(out_weights, keep_prob=self.keep_rate_output)) + out_biases
                self.unary_scores = tf.reshape(matricized_unary_scores, [tf.shape(self.sentences)[0], -1, n_classes[task_id]],
                                          name='unary_scores')

                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(self.unary_scores,
                                                                                      self.labels,
                                                                                      self.sentence_lens)
                self.transition_params = tf.identity(transition_params, name='transition_params')
                self.task_losses = tf.reduce_mean(-log_likelihood, name='task_logl_loss')
                # total_loss = tf.add(self.task_losses, self.domain_loss, name='total_loss')

                tv = tf.trainable_variables()
                self.regularization_cost = reg_coef * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv], name='regularization_cost')
                if adv_coef > 0.0:
                    total_loss = tf.add(tf.add(self.task_losses, adv_coef * self.domain_loss), self.regularization_cost,
                                        name='total_loss')
                else:
                    total_loss = tf.add(self.task_losses, self.regularization_cost,
                                        name='total_loss')






