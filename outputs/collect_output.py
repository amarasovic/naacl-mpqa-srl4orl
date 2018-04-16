import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='collect results over folds',
                                     add_help=False, conflict_handler='resolve')

    parser.add_argument('--begin_fold', type=int, default=0, help='start fold')
    parser.add_argument('--end_fold', type=int, default=10, help='end fold')
    parser.add_argument('--model', type=str, default=None, help='fs, sp, asp, wsp, hmtl')
    parser.add_argument('--exp_setup_id', type=str, default=None, help='prior or new')
    parser.add_argument('--seed', type=int, default=24, help='random seed')
    argv = parser.parse_args()

    out_dir_full = argv.exp_setup_id + '/' + str(argv.seed) + '/' + argv.model + '/orl/results/'
    all_folds_dev_file = open(out_dir_full + 'all_folds_dev_results.txt', 'w')
    all_folds_test_file = open(out_dir_full + 'all_folds_test_results.txt', 'w')
    all_checkpoints_file = open(out_dir_full + 'all_folds_checkpoints.txt', 'w')

    for fold in range(argv.begin_fold, argv.end_fold):
        res_file = open(out_dir_full + str(fold+1) + '/results.txt').readlines()

        all_folds_dev_file.write('\t'.join([str(float(x)*100) for x in res_file[0].split('\t')[1:5]]) + '\n')
        all_folds_test_file.write('\t'.join([str(float(x)*100) for x in res_file[1].split('\t')[1:5]]) + '\n')
        all_checkpoints_file.write('\t'.join([x for x in res_file[1].split('\t')[5:]]))

    '''
    out_dir_full = argv.model + '/srl/results/'
    all_folds_dev_file = open(out_dir_full + 'all_folds_dev_results.txt', 'w')
    all_folds_test_file = open(out_dir_full + 'all_folds_test_results.txt', 'w')
    all_checkpoints_file = open(out_dir_full + 'all_folds_checkpoints.txt', 'w')

    for fold in range(argv.begin_fold, argv.end_fold):
        res_file = open(out_dir_full + str(fold+1) + '/results.txt').readlines()

        all_folds_dev_file.write(str(float(res_file[0].split('\t')[3])*100) + '\n')
        all_folds_test_file.write(str(float(res_file[1].split('\t')[3])*100) + '\n')
        all_checkpoints_file.write('\t'.join([x for x in res_file[1].split('\t')[4:]]))
    '''










