#!/usr/bin/env bash

data_dir="data/NELL-995"
model="point.rs.conve"
group_examples_by_query="False"
use_action_space_bucketing="True"
use_conf="True"
support_times="True"
path_search_policy="tree"
baseline="curriculum"
decrease_step=8
decrease_rate=0.9
decrease_offline=0.6

bandwidth=256
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=3
num_epochs=200
num_wait_epochs=20
num_peek_epochs=2
bucket_interval=10
batch_size=128
train_batch_size=128
dev_batch_size=32
learning_rate=0.003
grad_norm=5
emb_dropout_rate=0.1
ff_dropout_rate=0.1
action_dropout_rate=0.1
action_dropout_anneal_interval=1000
reward_shaping_threshold=0
beta=0.05
relation_only="False"
beam_size=512

distmult_state_dict_path="model/"
complex_state_dict_path="model/NELL-995-complex-RV-xavier-200-200-0.003-0.3-0.1/model_best.tar"
conve_state_dict_path="model/NELL-995-conve-RV-xavier-200-200-0.003-32-3-0.3-0.3-0.2-0.1/model_best.tar"
checkpoint_path="model/NELL-995.test-point.rs.conve-xavier-curriculum-200-200-3-0.003-0.1-0.1-0.1-256-0.05-5.0-0.9-useConf-test/model_best.tar"

num_paths_per_entity=-1
margin=-1
