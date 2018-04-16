# load for asp, sp, fs
from train_utils import _train_task_ops, train_task_step
from eval_utils import eval_orl, eval_srl, record_results, plot_training_curve
import tensorflow as tf
import numpy as np
import logging
import time as ti
import os


def train(argv):
    train_iter = argv.train_iter
    srl_train_iter_eval = argv.srl_train_iter_eval
    orl_train_iter_eval = argv.orl_train_iter_eval
    srl_dev_iter = argv.srl_dev_iter
    srl_test_iter = argv.srl_test_iter
    orl_dev_iter = argv.orl_dev_iter
    orl_test_iter = argv.orl_test_iter
    label_dict_inv = argv.srl_label_dict_inv
    embeddings = argv.embeddings

    # allow only pre-tained embeddings at the moment
    assert isinstance(embeddings, np.ndarray)

    if argv.model in ['asp', 'sp']:
        logging.info('loading (a)sp model')
        from models.sp_mtl_models_v2 import SRL4ORL_deep_tagger

    if argv.model == 'fs':
        logging.info('loading fs model')
        from models.fs_mtl_model import SRL4ORL_deep_tagger

    if argv.model == 'hmtl':
        logging.info('loading hmtl model')
        from models.hmtl_model import SRL4ORL_deep_tagger

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(argv.seed)

        gpu_options = tf.GPUOptions(allow_growth=True, allocator_type='BFC',
                                    per_process_gpu_memory_fraction=argv.gpu_fraction)

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True,
                                      gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            seq_model = SRL4ORL_deep_tagger(argv.n_classes_srl,
                                            argv.n_classes_orl,
                                            embeddings,
                                            argv.embeddings_trainable,
                                            argv.hidden_size,
                                            argv.cell,
                                            argv.seed,
                                            argv.n_layers_shared,
                                            argv.n_layers_orl,
                                            argv.adv_coef,
                                            argv.reg_coef)

            train_task_ops = _train_task_ops(argv.lr, argv.grad_clip, argv.model)

            init_vars = tf.global_variables_initializer()
            sess.run(init_vars)

            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            logging.info(param_stats)

            # output directory for models
            timestamp = str(int(ti.time()))
            fname = 'runs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/' + argv.model + '/' + str(argv.fold+1) + '/'
            out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                                   fname,
                                                   timestamp))
            logging.info('writing to %s ' % out_dir)

            # checkpoint setup
            checkpoints_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_best = os.path.join(checkpoints_dir, 'model')
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)

            saver = tf.train.Saver(tf.global_variables())

            srl_fdev_best = 0.0
            orl_fdev_best = 0.0 # proportional

            srl_flist = [[], [], []]
            holder_flist = [[], [], []]
            target_flist = [[], [], []]

            train_iter_ti = 0.0
            train_count = 0

            early_stopping = [True]*argv.early_stop
            for it, batch in enumerate(train_iter):
                task_id = int(it % 2)

                start_iter = ti.time()
                _, _, dscrm_logits_tf = train_task_step(seq_model,
                                                        sess,
                                                        train_task_ops[task_id],
                                                        task_id,
                                                        batch,
                                                        argv.keep_rate_input,
                                                        argv.keep_rate_output,
                                                        argv.keep_state_rate,
                                                        argv.model,
                                                        )
                curr_time = ti.time()-start_iter
                train_iter_ti += curr_time
                train_count += len(batch)

                if (it+1) % argv.eval_every == 0:
                    avg_iter_time = train_iter_ti / (float((it + 1) * 60))
                    avg_epoch_time = train_iter_ti / float(((it + 1) / float(argv.eval_every)) * 60.0)
                    inst_per_sec = float(train_count) / float(train_iter_ti)
                    logging.info('fold[%s]-iter[%s]: avg-batch-train-time=%s, avg-epoch-train-time=%s, train-instances/s=%s' %
                                 (str(argv.fold+1), str(it+1), str(avg_iter_time), str(avg_epoch_time), str(inst_per_sec)))


                    '''
                    srl_eval_time = ti.time()
                    p_train, r_train, f_train = eval_srl(srl_train_iter_eval, label_dict_inv, sess, seq_model, 0)
                    p_dev, r_dev, f_dev = eval_srl(srl_dev_iter, label_dict_inv, sess, seq_model, 0)
                    p_test, r_test, f_test = eval_srl(srl_test_iter, label_dict_inv, sess, seq_model, 0)
                    curr_time = (ti.time()-srl_eval_time)/60.0
                    num_inst = sum([len(x) for x in srl_train_iter_eval]) + sum([len(x) for x in srl_dev_iter]) + \
                               sum([len(x) for x in srl_test_iter])
                    logging.info('fold[%s]-iter[%s]: eval-srl-time=%s, eval-srl-inst/s=%s' %
                                 (str(argv.fold+1), str(it+1), str(curr_time), str(curr_time / float(num_inst))))
                    logging.info('fold[%s]-iter[%s]: srl-f1-train=%s, srl-f1-dev=%s, srl-f1-test=%s, srl-eval-time=%s' %
                                 (str(argv.fold+1), str(it+1), str(f_train), str(f_dev), str(f_test), str(curr_time)))

                    srl_flist[0].append(f_train)
                    srl_flist[1].append(f_dev)
                    srl_flist[2].append(f_test)

                    fig_path = argv.out_dir + 'srl/figs/' + str(argv.fold + 1) + '/'
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)
                    plot_training_curve(fig_path + 'learning_curve.png', (it+1)/argv.eval_every, srl_flist)
                    

                    if f_dev > srl_fdev_best:
                        logging.info('better srl dev score!')
                        srl_fdev_best = f_dev
                        respath = argv.out_dir + 'srl/results/' + str(argv.fold + 1) + '/'
                        if not os.path.exists(respath):
                            os.makedirs(respath)

                        reslist = [argv.fold+1, p_dev, r_dev, f_dev, param_stats.total_parameters, avg_iter_time, inst_per_sec,  checkpoints_dir]
                        record_results(respath + 'results.txt', reslist, 'dev')

                        reslist = [argv.fold+1, p_test, r_test, f_test, param_stats.total_parameters, avg_iter_time, inst_per_sec,  checkpoints_dir]
                        record_results(respath + 'results.txt', reslist, 'test')
                    '''

                    orl_eval_time = ti.time()
                    binary_fscore_train, proportional_fscore_train = eval_orl(orl_train_iter_eval, sess, seq_model, 1)
                    binary_fscore_dev, proportional_fscore_dev = eval_orl(orl_dev_iter, sess, seq_model, 1)
                    binary_fscore_test, proportional_fscore_test = eval_orl(orl_test_iter, sess, seq_model, 1)
                    curr_time = (ti.time()-orl_eval_time)/60.0
                    num_inst = sum([len(x) for x in orl_train_iter_eval]) + sum([len(x) for x in orl_dev_iter]) + \
                               sum([len(x) for x in orl_test_iter])
                    logging.info('fold[%s]-iter[%s]: eval-orl-time=%s, eval-orl-inst/s=%s' %
                                 (str(argv.fold+1), str(it+1), str(curr_time), str(curr_time / float(num_inst))))

                    holder_flist[0].append(proportional_fscore_train[1])
                    holder_flist[1].append(proportional_fscore_dev[1])
                    holder_flist[2].append(proportional_fscore_test[1])

                    target_flist[0].append(proportional_fscore_train[2])
                    target_flist[1].append(proportional_fscore_dev[2])
                    target_flist[2].append(proportional_fscore_test[2])

                    logging.info('fold[%s]-iter[%s]: holder-bin-f1-train=%s, holder-bin-f1-dev=%s, holder-bin-f1-test=%s' %
                                 (str(argv.fold+1), str(it+1), str(binary_fscore_train[1]), str(binary_fscore_dev[1]),
                                  str(binary_fscore_test[1])))
                    logging.info('fold[%s]-iter[%s]: target-bin-f1-train=%s, target-bin-f1-dev=%s, target-bin-f1-test=%s' %
                                 (str(argv.fold+1), str(it+1), str(binary_fscore_train[2]), str(binary_fscore_dev[2]),
                                  str(binary_fscore_test[2])))

                    logging.info('fold[%s]-iter[%s]: holder-prop-f1-train=%s, holder-prop-f1-dev=%s, holder-prop-f1-test=%s' % (
                        str(argv.fold + 1), str(it+1), str(proportional_fscore_train[1]), str(proportional_fscore_dev[1]),
                        str(proportional_fscore_test[1])))
                    logging.info('fold[%s]-iter[%s]: target-prop-f1-train=%s, target-prop-f1-dev=%s, target-prop-f1-test=%s' % (
                        str(argv.fold + 1), str(it+1), str(proportional_fscore_train[2]), str(proportional_fscore_dev[2]),
                        str(proportional_fscore_test[2])))

                    fig_path = argv.out_dir + 'orl/figs/holder/' + str(argv.fold + 1) + '/'
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)
                    plot_training_curve(fig_path + 'learning_curve.png', (it+1)/argv.eval_every, holder_flist)

                    fig_path = argv.out_dir + 'orl/figs/target/' + str(argv.fold + 1) + '/'
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)
                    plot_training_curve(fig_path + 'learning_curve.png', (it+1)/argv.eval_every, target_flist)

                    early_stopping.pop(0)
                    early_stopping.append(False)
                    if np.mean(np.asarray(proportional_fscore_dev[1:])) > orl_fdev_best:
                        logging.info('better orl dev score!')
                        early_stopping[-1] = True
                        orl_fdev_best = np.mean(np.asarray(proportional_fscore_dev[1:]))
                        #respath = argv.out_dir + 'orl/results/' + str(argv.fold + 1) + '/results.txt'
                        respath = argv.out_dir + 'orl/results/' + str(argv.fold + 1) + '/'
                        if not os.path.exists(respath):
                            os.makedirs(respath)

                        reslist = [argv.fold+1, binary_fscore_dev[1], proportional_fscore_dev[1],
                                                binary_fscore_dev[2], proportional_fscore_dev[2],
                                   param_stats.total_parameters, avg_iter_time, inst_per_sec, checkpoints_dir]
                        record_results(respath + 'results.txt', reslist, 'dev')

                        reslist = [argv.fold+1, binary_fscore_test[1], proportional_fscore_test[1],
                                                binary_fscore_test[2], proportional_fscore_test[2],
                                   param_stats.total_parameters, avg_iter_time, inst_per_sec, checkpoints_dir]
                        record_results(respath + 'results.txt', reslist, 'test')

                        # save
                        path = saver.save(sess, checkpoint_best)
                        logging.info('saved best model checkpoint to {}\n'.format(path))

                    if True not in early_stopping:
                        break


