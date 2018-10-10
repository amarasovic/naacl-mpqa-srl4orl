# SRL4ORL: Improving Opinion Role Labeling Using Multi-Task Learning With Semantic Role Labeling
This repository contains code for reproducing experiments done in:

Ana Marasovic and Anette Frank (2018): [SRL4ORL: Improving Opinion Role Labeling Using Multi-Task Learning With Semantic Role Labeling](https://arxiv.org/abs/1711.00768). In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT). New Orleans, USA.


## Requirements

- [tensorflow 1.4](https://www.tensorflow.org/versions/r1.4/)
- [nltk](http://www.nltk.org)
- [gensim](https://radimrehurek.com/gensim/)
- [matplotlib](https://matplotlib.org)
- [scikit-learn](http://scikit-learn.org/stable/)

## Data

### ORL

Download [MPQA 2.0 corpus](http://mpqa.cs.pitt.edu/corpora/mpqa_corpus/mpqa_corpus_2_0/).

<span style="color:red">Check ```mpqa2-pytools``` for example usage.</span>


Splits can be found in the ```datasplit``` folder.

### SRL 

The data is provided by: [CoNLL-2005 Shared Task](http://www.lsi.upc.edu/~srlconll/soft.html), but the original words are from the Penn Treebank dataset, which is not publicly available.

## How to train models?

```

python main.py --adv_coef 0.0 --model fs --exp_setup_id new --n_layers_orl 0 --begin_fold 0 --end_fold 4

python main.py --adv_coef 0.0 --model html --exp_setup_id new --n_layers_orl 1 --n_layers_shared 2 --begin_fold 0 --end_fold 4

python main.py --adv_coef 0.0 --model sp --exp_setup_id new --n_layers_orl 3 --begin_fold 0 --end_fold 4

python main.py --adv_coef 0.1 --model asp --exp_setup_id prior --n_layers_orl 3 --begin_fold 0 --end_fold 10

```

## Reference

If you make use of the contents of this repository, please cite [the following paper](http://aclweb.org/anthology/N18-1054):

```
@inproceedings{marasovicfrank:srl4orl,
  title={{Improving Opinion Role Labeling Using Multi-Task Learning With Semantic Role Labeling}},
  author={Marasovi\'{c}, Ana and Frank, Anette},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
  year={2018},
  address={New Orleans, USA},
  note={to appear}
}
```
