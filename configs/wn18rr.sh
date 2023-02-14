#!/usr/bin/env bash

data_dir="data/WN18RR"
model="point"
group_examples_by_query="False"
use_action_space_bucketing="True"

bandwidth=500
entity_dim=200
relation_dim=200
history_dim=200
history_num_layers=3
num_rollouts=20
num_rollout_steps=4
bucket_interval=10
num_epochs=100
num_wait_epochs=20
num_peek_epochs=1
batch_size=512
train_batch_size=512
dev_batch_size=64
learning_rate=0.001
baseline="n/a"
grad_norm=0
emb_dropout_rate=0.1
ff_dropout_rate=0.1
action_dropout_rate=0.1
action_dropout_anneal_interval=1000
beta=0
relation_only="False"
beam_size=128
checkpoint_path="model/WN18RR-point-xavier-n/a-200-200-3-0.001-0.1-0.1-0.1-500-0.0-4-support_not_times/checkpoint-19.tar"

num_paths_per_entity=-1
margin=-1
