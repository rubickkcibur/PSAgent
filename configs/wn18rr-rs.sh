#!/usr/bin/env bash

data_dir="data/WN18RR"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"
support_times="True"
path_search_policy="tree"
decrease_step=0
decrease_rate=0
decrease_offline=0

bandwidth=500
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=5
bucket_interval=10
num_epochs=1000
num_wait_epochs=20
num_peek_epochs=2
batch_size=512
train_batch_size=512
dev_batch_size=128
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.1
ff_dropout_rate=0.1
action_dropout_rate=0.1
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0
relation_only="False"
beam_size=128

distmult_state_dict_path="model/WN18RR-distmult-xavier-200-200-0.003-0.2-0.1/model_best.tar"
complex_state_dict_path="model/WN18RR-complex-RV-xavier-200-200-0.003-0.2-0.1/model_best.tar"
conve_state_dict_path="model/WN18RR-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
checkpoint_path="None"

num_paths_per_entity=-1
margin=-1