"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Compute Evaluation Metrics.
 Code adapted from https://github.com/TimDettmers/ConvE/blob/master/evaluation.py
"""

import numpy as np
import pickle
from tqdm import tqdm

import torch

from src.parse_args import args
from src.data_utils import NO_OP_ENTITY_ID, DUMMY_ENTITY_ID
from src.rules.rule import HornRule
from src.utils.ops import format_path
from src.utils.vis import visualize_two_array


def hits_and_ranks(examples, scores, search_traces, kg, verbose=False, where="test"):
    """
    Compute ranking based metrics.
    """
    all_answers = kg.all_objects if where == "test" else kg.dev_objects
    # print(len(examples),len(scores))
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    relations = []
    dev_metrics = {}
    for i, example in enumerate(examples):
        e1, e2, r = example
        relations.append(r)
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score
        # print('e2_multi',e2_multi)
        # print('scores',scores[i])
        # print('scores[i][e2]',scores[i][e2])

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()
    # print('top_k_scores',top_k_scores)
    # print('top_k_targets',top_k_targets)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    mrr = 0
    q_r = []
    poses = []
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])

        # pos1 = np.where(top_k_targets[i] == e2)[0]
        pos = np.asarray([j in e2_multi for j in top_k_targets[i]]).nonzero()[0]

        if len(pos) > 0:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1
            mrr += 1.0 / (pos + 1)
            q_r.append(r)
            poses.append(pos)
        else:
            q_r.append(r)
            poses.append(-1)

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)
    mrr = float(mrr) / len(examples)

    if verbose:
        print('Hits@1 = {:.3f}'.format(hits_at_1))
        print('Hits@3 = {:.3f}'.format(hits_at_3))
        print('Hits@5 = {:.3f}'.format(hits_at_5))
        print('Hits@10 = {:.3f}'.format(hits_at_10))
        print('MRR = {:.3f}'.format(mrr))
        # visualize_two_array(poses,[kg.r2attr[kg.id2relation[r]][0] for r in q_r],"src/pos_func.png")
    dev_metrics["hit1"] = hits_at_1
    dev_metrics["hit3"] = hits_at_3
    dev_metrics["hit5"] = hits_at_5
    dev_metrics["hit10"] = hits_at_10
    dev_metrics["mrr"] = mrr
    dev_metrics["conf1"] = -1
    dev_metrics["conf2"] = -1

    """
    computing confidence
    """
    if not search_traces is None:
        max_paths = []
        for i in range(len(examples)):
            pos = poses[i]
            if pos >= 0:
                j = top_k_targets[i][pos]
                trace = search_traces[i][j]
                max_paths.append(trace)
            else:
                max_paths.append([])

        confs = []
        print("computing confs")
        for i in tqdm(range(len(max_paths))):
            search_trace = max_paths[i]
            query_r = relations[i]
            pos = poses[i]
            if pos < 0 or len(search_trace) <= 0:
                continue
            rule = HornRule(path=search_trace,head=(search_trace[0][1],query_r,search_trace[-1][1]),mode="CONSTANT_HEAD")
            if rule.get_str_representation() in kg.store_eval_conf:
                confs.append(kg.store_eval_conf[rule.get_str_representation()])
            else:
                path_trees = kg.cnt_paths_by_rule(rule,[search_trace[0][1]],where="all")
                conf_e1 = []
                conf_e2 = []
                conf_r = []
                conf_pe2 = []
                times = []
                for (pe2,weight) in path_trees.items():
                    # for j in range(len(leaves)):
                    #     if leaves[j] <= 0:
                    #         continue
                    conf_e1.append(search_trace[0][1])
                    conf_e2.append(search_trace[-1][1])
                    conf_r.append(query_r)
                    conf_pe2.append(pe2)
                    times.append(weight) 
                rewards = []
                for j in range(len(conf_e1)):
                    rewards.append(1 if kg.in_graph(conf_e1[j],conf_r[j],conf_pe2[j],"all") else 0)
                rewards = np.array(rewards)
                # if len(rewards) == 0:
                #     print(search_trace)
                conf1 = np.sum(rewards*times)/(np.sum(times)+1e-6)
                conf2 = np.sum(rewards)/(len(rewards)+1e-6) 
                confs.append((conf1,conf2))
                kg.store_eval_conf[rule.get_str_representation()] = (conf1,conf2)
        c1,c2 = path_pression(confs)
        dev_metrics["conf1"] = c1
        dev_metrics["conf2"] = c2
        # with open("src/pos.txt","w") as f:
        #     for i in range(len(poses)):
        #         if poses[i] < 0:
        #             continue
        #         f.write("{}\t{}\t{}\t{}\t{}\n".format(
        #             kg.id2relation[q_r[i]],
        #             poses[i],
        #             confs[i][1],
        #             kg.r2attr[kg.id2relation[q_r[i]]][5],
        #             format_path(max_paths[i],kg)
        #         ))
    return dev_metrics

def path_pression(pred_confs):
    conf1s = [t[0] for t in pred_confs]
    conf2s = [t[1] for t in pred_confs]
    conf1 = sum(conf1s)/len(conf1s)
    conf2 = sum(conf2s)/len(conf2s)
    print('IMPS = {:.4f}'.format(conf1))
    return conf1,conf2

def hits_at_k(examples, scores, all_answers, verbose=False):
    """
    Hits at k metrics.
    :param examples: List of triples and labels (+/-).
    :param pred_targets:
    :param scores:
    :param all_answers:
    :param verbose:
    """
    assert(len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = list(all_answers[e1][r]) + dummy_mask
        # save the relevant prediction
        target_score = scores[i, e2]
        # mask all false negatives
        scores[i][e2_multi] = 0
        scores[i][dummy_mask] = 0
        # write back the save prediction
        scores[i][e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if pos:
            pos = pos[0]
            if pos < 10:
                hits_at_10 += 1
                if pos < 5:
                    hits_at_5 += 1
                    if pos < 3:
                        hits_at_3 += 1
                        if pos < 1:
                            hits_at_1 += 1

    hits_at_1 = float(hits_at_1) / len(examples)
    hits_at_3 = float(hits_at_3) / len(examples)
    hits_at_5 = float(hits_at_5) / len(examples)
    hits_at_10 = float(hits_at_10) / len(examples)

    if verbose:
        print('Hits@1 = {:.3f}'.format(hits_at_1))
        print('Hits@3 = {:.3f}'.format(hits_at_3))
        print('Hits@5 = {:.3f}'.format(hits_at_5))
        print('Hits@10 = {:.3f}'.format(hits_at_10))
    return hits_at_1, hits_at_3, hits_at_5, hits_at_10

def hits_and_ranks_by_seen_queries(examples, scores, all_answers, seen_queries, verbose=False):
    seen_exps, unseen_exps = [], []
    seen_ids, unseen_ids = [], []
    for i, example in enumerate(examples):
        e1, e2, r = example
        if (e1, r) in seen_queries:
            seen_exps.append(example)
            seen_ids.append(i)
        else:
            unseen_exps.append(example)
            unseen_ids.append(i)

    _, _, _, _, seen_mrr = hits_and_ranks(seen_exps, scores[seen_ids], all_answers, verbose=False)
    _, _, _, _, unseen_mrr = hits_and_ranks(unseen_exps, scores[unseen_ids], all_answers, verbose=False)
    if verbose:
        print('MRR on seen queries: {:.3f}'.format(seen_mrr))
        print('MRR on unseen queries: {:.3f}'.format(unseen_mrr))
    return seen_mrr, unseen_mrr

def hits_and_ranks_by_relation_type(examples, scores, all_answers, relation_by_types, verbose=False):
    to_M_rels, to_1_rels = relation_by_types
    to_M_exps, to_1_exps = [], []
    to_M_ids, to_1_ids = [], []
    for i, example in enumerate(examples):
        e1, e2, r = example
        if r in to_M_rels:
            to_M_exps.append(example)
            to_M_ids.append(i)
        else:
            to_1_exps.append(example)
            to_1_ids.append(i)

    _, _, _, _, to_m_mrr = hits_and_ranks(to_M_exps, scores[to_M_ids], all_answers, verbose=False)
    _, _, _, _, to_1_mrr = hits_and_ranks(to_1_exps, scores[to_1_ids], all_answers, verbose=False)
    if verbose:
        print('MRR on to-M relations: {:.3f}'.format(to_m_mrr))
        print('MRR on to-1 relations: {:.3f}'.format(to_1_mrr))
    return to_m_mrr, to_1_mrr

def link_MAP(examples, scores, labels, all_answers, verbose=False):
    """
    Per-query mean average precision.
    """
    assert (len(examples) == len(scores))
    queries = {}
    for i, example in enumerate(examples):
        e1, e2, r = example
        if not e1 in queries:
            queries[e1] = []
        queries[e1].append((examples[i], labels[i], scores[i][e2]))

    aps = []
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]

    for e1 in queries:
        ranked_examples = sorted(queries[e1], key=lambda x:x[2], reverse=True)
        acc_precision, offset, num_pos = 0, 0, 0
        for i in range(len(ranked_examples)):
            triple, label, score = ranked_examples[i]
            _, r, e2 = triple
            if label == '+':
                num_pos += 1
                acc_precision += float(num_pos) / (i + 1 - offset)
            else:
                answer_set = {}
                if e1 in all_answers and r in all_answers[e1]:
                    answer_set = all_answers[e1][r]
                if e2 in answer_set or e2 in dummy_mask:
                    print('False negative found: {}'.format(triple))
                    offset += 1
        if num_pos > 0:
            ap = acc_precision / num_pos
            aps.append(ap)
    map = np.mean(aps)
    if verbose:
        print('MAP = {:.3f}'.format(map))
    return map

def export_error_cases(examples, scores, all_answers, output_path):
    """
    Export indices of examples to which the top-1 prediction is incorrect.
    """
    assert (len(examples) == scores.shape[0])
    # mask false negatives in the predictions
    dummy_mask = [DUMMY_ENTITY_ID, NO_OP_ENTITY_ID]
    for i, example in enumerate(examples):
        e1, e2, r = example
        e2_multi = dummy_mask + list(all_answers[e1][r])
        # save the relevant prediction
        target_score = float(scores[i, e2])
        # mask all false negatives
        scores[i, e2_multi] = 0
        # write back the save prediction
        scores[i, e2] = target_score

    # sort and rank
    top_k_scores, top_k_targets = torch.topk(scores, min(scores.size(1), args.beam_size))
    top_k_targets = top_k_targets.cpu().numpy()

    top_1_errors, top_10_errors = [], []
    for i, example in enumerate(examples):
        e1, e2, r = example
        pos = np.where(top_k_targets[i] == e2)[0]
        if len(pos) <= 0 or pos[0] > 0:
            top_1_errors.append(i)
        if len(pos) <= 0 or pos[0] > 9:
            top_10_errors.append(i)
    with open(output_path, 'wb') as o_f:
        pickle.dump([top_1_errors, top_10_errors], o_f)

    print('{}/{} top-1 error cases written to {}'.format(len(top_1_errors), len(examples), output_path))
    print('{}/{} top-10 error cases written to {}'.format(len(top_10_errors), len(examples), output_path))

