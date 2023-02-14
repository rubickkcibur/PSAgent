# Path Spuriousness-aware Reinforcement Learning for Multi-Hop Knowledge Graph Reasoning

Our implementation is based on official codes of [MultiHop](https://github.com/salesforce/MultiHopKG). Best regards to their contribution.

## Quick Start

#### Mannually set up 
```
conda install --yes --file requirements.txt
```

### Evaluate pre-trained models
Taking umls as example, if you want model performance of MultiHop,
change `checkpoint_path` item to `umls_multihop.tar` in `umls-rs.sh` and then run the following command:
```
./experiment-rs.sh configs/umls-rs.sh --inference <gpu-ID> --save_beam_search_paths
```
If you want model performance of Ours+min,
change `checkpoint_path` item to `umls_min.tar` in `umls-conf.sh` and then run the following command:
```
./experiment-rs.sh configs/umls-conf.sh --inference <gpu-ID> --save_beam_search_paths
```
If you want model performance of Ours+sum,
change `checkpoint_path` item to `umls_sum.tar` in `umls-conf.sh` and then run the following command:
```
./experiment-rs.sh configs/umls-conf.sh --inference <gpu-ID> --save_beam_search_paths
```

### Process data
You can fetch `umls`, `kinship`, `wn18rr`, `nell-995` from [MultiHop](https://github.com/salesforce/MultiHopKG).

Then run the following command to preprocess the datasets.
```
./experiment.sh configs/<dataset>.sh --process_data <gpu-ID>
```

`<dataset>` is the name of any dataset folder in the `./data` directory.

### Train models
Then the following commands can be used to train the proposed models and baselines in the paper. By default, dev set evaluation results will be printed when training terminates.

1. Train embedding-based models
```
./experiment-emb.sh configs/<dataset>-<emb_model>.sh --train <gpu-ID>
```
The following embedding-based models are implemented: `distmult`, `complex` and `conve`.

2. Train RL models (policy gradient)
```
./experiment.sh configs/<dataset>.sh --train <gpu-ID> --save_beam_search_paths
```

3. Train RL models (policy gradient + reward shaping)
```
./experiment-rs.sh configs/<dataset>-rs.sh --train <gpu-ID> --save_beam_search_paths
```

4. Train PS-aware RL models
```
./experiment-rs.sh configs/<dataset>-conf.sh --train <gpu-ID> --save_beam_search_paths
```

* Note: To train the RL models using reward shaping, make sure 1) you have pre-trained the embedding-based models and 2) set the file path pointers to the pre-trained embedding-based models correctly ([example configuration file](configs/umls-rs.sh)).

* Note: `--save_beam_search_paths` flag indicates you want an evaluation on IMPS metric and it is not applicable for embedding models because they have no reasoning paths 

* Note for the NELL-995 dataset: 

  On this dataset we split the original training data into `train.triples` and `dev.triples`, and the final model to test has to be trained with these two files combined. 
  1. To obtain the correct test set results, you need to add the `--test` flag to all data pre-processing, training and inference commands.  
    ```
    # You may need to adjust the number of training epochs based on the dev set development.

    ./experiment.sh configs/nell-995.sh --process_data <gpu-ID> --test
    ./experiment-emb.sh configs/nell-995-conve.sh --train <gpu-ID> --test
    ./experiment-rs.sh configs/NELL-995-conf.sh --train <gpu-ID> --test
    ./experiment-rs.sh configs/NELL-995-conf.sh --inference <gpu-ID> --test
    ```    
  2. Leave out the `--test` flag during development.

### Change the hyperparameters
To change the hyperparameters and other experiment set up, start from the [configuration files](configs).
