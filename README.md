
# Not All Relevance Scores are Equal: Efficient Uncertainty and Calibration Modeling for Deep Retrieval Models
_SIGIR'21 Daniel Cohen, Bhaskar Mitra, Oleg Lesota, Navid Rekabsaz, and Carsten Eickhoff_, [arXiv](https://arxiv.org/abs/2105.04651)

This repository contains a framework intended for investigating the impact of last layer(s) bayesian modeling on deep retrieval architectures.
To prepare the training data, please consult the "How to train the models" section [in this repository](https://github.com/sebastian-hofstaetter/sigir19-neural-ir).


## Recommended setup

* Python 3.7+;
* Pytorch 1.3+;

PIP:
* Anaconda;
* Pytorch 1.3+;
* Pip (Anaconda);
* Allennlp (pip) version 0.9.0;
* Gensim;
* GPUtil;
* pip install transformers ;

To install, navigate to repository directory and (after activating the environment) run:
`pip install -r requirements.txt`

## General Usage
1) In configs/msmarco-passge.yaml set the needed work folders (```expirement_base_path:```, ```debug_base_path:```), this is were the <RUN_FOLDERS> of your experiments will be located;
2) python3 train.py --run-name experiment1 --config-file configs/msmarco-passage.yaml --cuda --gpu-id 0

### Usage - test
```sh
$ python3 train.py --cuda --gpu-id 0 --run-folder <RUN_FOLDER> --test
```
* key ```--custom-test-depth <int>``` to fix reranking depth during test;
* key ```--test-files-prefix <str>``` to add meaningful pre to saved test files. Files do not get overwrited. Meaningless prefixes are added in case of conflicts.

Set those three below for new custom test set:
* key ```--custom-test-tsv <str>```
* key ```--custom-test-qrels <str>```
* key ```--custom-test-candidates <str>```
  
To set the number of monte carlo samples for bayesian models:
* key ```--custom-test-num-models <num>``` * Note, set 1 for non-bayesian models. During training, the code will overwrite to 1 to avoid unnecessary compute time.

### Usage - test commands:
Test-set 1: **SPARSE**
```sh
$ python3 train.py --cuda --test --custom-test-depth 200 --custom-test-tsv "<...>/validation.not-subset.top200.cleaned.split-4/*" --custom-test-qrels "/share/cp/datasets/ir/msmarco/passage/qrels.dev.tsv" --custom-test-candidates "/share/cp/datasets/ir/msmarco/passage/run.msmarco-passage.BM25_k1_0.9_b_0.4.dev.txt" --test-files-pretfix "SPARSE-" --run-folder <run_folder> --gpu-id 0
```

Test-set 2: **TREC - 2019**
```sh
$ python3 train.py --cuda --test --custom-test-depth 200 --custom-test-tsv "<...>/test2019.top1000.cleaned.split-4/*" --custom-test-qrels "/share/cp/datasets/ir/msmarco/passage/test2019-qrels.txt" --custom-test-candidates "/share/cp/datasets/ir/msmarco/passage/run.msmarco-passage.BM25-k1_0.82_b_0.72.test2019.txt" --test-files-pretfix "TREC-19-" --run-folder <run_folder> --gpu-id 0
```


## Other
* key ```--debug``` can be used to check if the whole pipeline is in one piece: it shortens training, validation and test;
* make sure to adjust "expirement_base_path" and "debug_base_path" in ```configs/*.yaml```

## "Best" settings for running different models
* Conv-KNRM
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: conv_knrm, loss: maxmargin, param_group0_learning_rate: 0.001"
```

* Bert Mini/Tiny
```sh
$ python train.py --config-file [PATH_TO_CONFIG_FILE] --cuda --gpu-id 0 --config-overwrites "model: discbert, param_group0_learning_rate: 0.00003, token_embedder_type: bert, loss: crossentropy" --run-name DiscBert_Tiny
```

This repository was branched from [DeepGenIR](https://github.com/CPJKU/DeepGenIR) and developed to incorporate Concrete Monte Carlo Dropout and stochastic sampling. **The two repositories share the same structure and data preparation routine.**

The repository does not contain the code of the model used for cut-off prediction. Consult the original paper for more information about [Choppy](https://dl.acm.org/doi/10.1145/3397271.3401188).


