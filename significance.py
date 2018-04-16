import argparse

from scipy import stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='significance testing',
                                     add_help=False, conflict_handler='resolve')

    parser.add_argument('--seed', type=int, default=24, help='random seed')
    parser.add_argument('--seed1', type=int, default=24, help='random seed')
    parser.add_argument('--seed2', type=int, default=128, help='random seed')
    parser.add_argument('--exp_setup_id', type=str, default=None, help='prior or new')
    argv = parser.parse_args()

    if argv.exp_setup_id == 'prior':
        models = ['fs', 'hmtl', 'sp', 'asp']

        # significance between MTL and STL models
        for mode in ['dev', 'test']:
            sign_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/sign_stl_mtl_' + mode + '.txt', 'w')
            single_file = open('single-orl/outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
            stl_dstrs = [[] for _ in range(4)]

            for line in single_file:
                for i in range(4):
                    stl_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            for model in models:
                model_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/' + model + '/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                model_dstrs = [[] for _ in range(4)]
                for line in model_file:
                    for i in range(4):
                        model_dstrs[i].append(line.split('\t')[i].split('\n')[0])

                pvalues = []
                for i in range(4):
                    _, p = stats.ks_2samp(stl_dstrs[i], model_dstrs[i])
                    pvalues.append(p)

                conditions = ['sign' if p < 0.05 else 'not_sign' for p in pvalues]

                sign_file.write('\t'.join(conditions) + '\n')

        # significance between FS and other MTL models
        for mode in ['dev', 'test']:
            sign_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/sign_fs_mtl_' + mode + '.txt', 'w')
            fs_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/fs/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
            fs_dstrs = [[] for _ in range(4)]

            for line in fs_file:
                for i in range(4):
                    fs_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            for model in models[1:]:
                model_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/' + model + '/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                model_dstrs = [[] for _ in range(4)]
                for line in model_file:
                    for i in range(4):
                        model_dstrs[i].append(line.split('\t')[i].split('\n')[0])

                pvalues = []
                for i in range(4):
                    _, p = stats.ks_2samp(fs_dstrs[i], model_dstrs[i])
                    pvalues.append(p)

                conditions = ['sign' if p < 0.05 else 'not_sign' for p in pvalues]
                sign_file.write('\t'.join(conditions) + '\n')


        # significance between ASP and SP model
        for mode in ['dev', 'test']:
            sign_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/sign_asp_sp_' + mode + '.txt', 'w')
            asp_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/asp/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
            asp_dstrs = [[] for _ in range(4)]

            for line in asp_file:
                for i in range(4):
                    asp_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            sp_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/sp/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
            sp_dstrs = [[] for _ in range(4)]
            for line in sp_file:
                for i in range(4):
                    sp_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            pvalues = []
            for i in range(4):
                _, p = stats.ks_2samp(asp_dstrs[i], sp_dstrs[i])
                pvalues.append(p)

            conditions = ['sign' if p < 0.05 else 'not_sign' for p in pvalues]
            sign_file.write('\t'.join(conditions) + '\n')


    if argv.exp_setup_id == 'new':
        models = ['fs', 'hmtl', 'sp', 'asp']

        # significance between MTL and STL models
        for mode in ['dev', 'test']:
            sign_file = open('outputs/' + argv.exp_setup_id + '/sign_stl_mtl_' + mode + '.txt', 'w')
            single_file = []

            for seed in [argv.seed1, argv.seed2]:
                temp_file = open('single-orl/outputs/' + argv.exp_setup_id + '/' + str(seed) + '/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                single_file.extend(temp_file)

            stl_dstrs = [[] for _ in range(4)]

            for line in single_file:
                for i in range(4):
                    stl_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            for model in models:
                model_file = []
                for seed in [argv.seed1, argv.seed2]:
                    temp_file = open('outputs/' + argv.exp_setup_id + '/' + str(seed) + '/' + model + '/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                    model_file.extend(temp_file)

                model_dstrs = [[] for _ in range(4)]
                for line in model_file:
                    for i in range(4):
                        model_dstrs[i].append(line.split('\t')[i].split('\n')[0])

                pvalues = []
                for i in range(4):
                    _, p = stats.ks_2samp(stl_dstrs[i], model_dstrs[i])
                    pvalues.append(p)

                conditions = ['sign' if p < 0.05 else 'not_sign' for p in pvalues]

                sign_file.write('\t'.join(conditions) + '\n')


        # significance between FS and other MTL models
        for mode in ['dev', 'test']:
            sign_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/sign_fs_mtl_' + mode + '.txt', 'w')

            fs_file = []
            for seed in [argv.seed1, argv.seed2]:
                temp_file = open('outputs/' + argv.exp_setup_id + '/' + str(seed) + '/fs/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                fs_file.extend(temp_file)

            fs_dstrs = [[] for _ in range(4)]

            for line in fs_file:
                for i in range(4):
                    fs_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            for model in models[1:]:
                model_file = []
                for seed in [argv.seed1, argv.seed2]:
                    temp_file = open('outputs/' + argv.exp_setup_id + '/' + str(seed) + '/' + model + '/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                    model_file.extend(temp_file)

                model_dstrs = [[] for _ in range(4)]
                for line in model_file:
                    for i in range(4):
                        model_dstrs[i].append(line.split('\t')[i].split('\n')[0])

                pvalues = []
                for i in range(4):
                    _, p = stats.ks_2samp(fs_dstrs[i], model_dstrs[i])
                    pvalues.append(p)

                conditions = ['sign' if p < 0.05 else 'not_sign' for p in pvalues]
                sign_file.write('\t'.join(conditions) + '\n')


        # significance between ASP and SP model
        for mode in ['dev', 'test']:
            sign_file = open('outputs/' + argv.exp_setup_id + '/' + str(argv.seed) + '/sign_asp_sp_' + mode + '.txt', 'w')

            asp_file = []
            for seed in [argv.seed1, argv.seed2]:
                temp_file = open('outputs/' + argv.exp_setup_id + '/' + str(seed) + '/asp/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                asp_file.extend(temp_file)

            asp_dstrs = [[] for _ in range(4)]

            for line in asp_file:
                for i in range(4):
                    asp_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            sp_file = []
            for seed in [argv.seed1, argv.seed2]:
                temp_file = open('outputs/' + argv.exp_setup_id + '/' + str(seed) + '/sp/orl/results/all_folds_' + mode + '_results.txt', 'r').readlines()
                sp_file.extend(temp_file)

            sp_dstrs = [[] for _ in range(4)]
            for line in sp_file:
                for i in range(4):
                    sp_dstrs[i].append(line.split('\t')[i].split('\n')[0])

            pvalues = []
            for i in range(4):
                _, p = stats.ks_2samp(asp_dstrs[i], sp_dstrs[i])
                pvalues.append(p)

            conditions = ['sign' if p < 0.05 else 'not_sign' for p in pvalues]
            sign_file.write('\t'.join(conditions) + '\n')

