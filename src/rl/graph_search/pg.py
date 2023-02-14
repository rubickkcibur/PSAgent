"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Policy gradient (REINFORCE algorithm) training and inference.
"""

from cmath import isnan
import os
from statistics import mode
import numpy as np
from sys import path_hooks
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.rules.rule import HornRule

from src.learn_framework import LFramework
import src.rl.graph_search.beam_search as search
import src.utils.ops as ops
from src.utils.ops import int_fill_var_cuda, var_cuda, var_to_numpy, zeros_var_cuda
from src.utils.debug_store import conf_debug



class PolicyGradient(LFramework):
    def __init__(self, args, kg, pn):
        super(PolicyGradient, self).__init__(args, kg, pn)

        # Training hyperparameters
        self.relation_only = args.relation_only
        self.use_action_space_bucketing = args.use_action_space_bucketing
        self.num_rollouts = args.num_rollouts
        self.num_rollout_steps = args.num_rollout_steps
        self.baseline = args.baseline
        self.beta = args.beta  # entropy regularization parameter
        self.gamma = args.gamma  # shrinking factor
        self.action_dropout_rate = args.action_dropout_rate
        self.action_dropout_anneal_factor = args.action_dropout_anneal_factor
        self.action_dropout_anneal_interval = args.action_dropout_anneal_interval

        # Inference hyperparameters
        self.beam_size = args.beam_size
        self.use_conf = args.use_conf
        self.support_times = args.support_times

        # Analysis
        self.path_types = dict()
        self.num_path_types = 0
        self.rule2conf = {}

        #expert line
        self.ips_threshold = args.ips_threshold

    def reward_fun(self, e1, r, e2, pred_e2):
        return (pred_e2 == e2).float()

    def print_path(self,path_trace,ids):
        paths = [[] for i in range(len(path_trace[0][0]))]
        for step in range(len(path_trace)):
            for idx in range(len(path_trace[0][0])):
                paths[idx].append((path_trace[step][0][idx],path_trace[step][1][idx]))
        for id in ids:
            path_format = ops.format_path(paths[id],self.kg)
            print(path_format)

    def conf_fun(self,path_trace,e1,r,e2):
        paths = [[] for i in range(len(path_trace[0][0]))]
        confs = []
        for step in range(len(path_trace)):
            for idx in range(len(path_trace[0][0])):
                paths[idx].append((path_trace[step][0][idx],path_trace[step][1][idx]))
        lazy_load_cnt = 0
        for idx in range(len(paths)):
            path = paths[idx]
            rule = HornRule(path,(int(e1[idx]),int(r[idx]),path[-1][1]),mode="CONSTANT_HEAD")
            if rule.get_str_representation() in self.rule2conf:
                conf = self.rule2conf[rule.get_str_representation()]
                lazy_load_cnt += 1
            else:
                path_trees = self.kg.cnt_paths_by_rule(rule,[int(e1[idx])])
                conf_e1 = []
                conf_e2 = []
                conf_r = []
                conf_pe2 = []
                times = []
                for (pe2,weight) in path_trees.items():
                    # for i in range(len(leaves)):
                    #     if leaves[i] <= 0:
                    #         continue
                    conf_e1.append(int(e1[idx]))
                    conf_e2.append(int(e2[idx]))
                    conf_r.append(int(r[idx]))
                    conf_pe2.append(pe2)
                    if self.support_times:
                        times.append(weight)
                    else:
                        times.append(1)
                rewards = self.reward_fun(
                    var_cuda(torch.LongTensor(conf_e1), requires_grad=False),
                    var_cuda(torch.LongTensor(conf_r), requires_grad=False),
                    var_cuda(torch.LongTensor(conf_e2), requires_grad=False),
                    var_cuda(torch.LongTensor(conf_pe2), requires_grad=False)
                )
                rewards = var_to_numpy(rewards)
                conf = np.sum(rewards*times)/(np.sum(times)+1e-6)  
                self.rule2conf[rule.get_str_representation()] = conf
            confs.append(conf)
        return var_cuda(torch.FloatTensor(confs), requires_grad=False)

    def loss(self, mini_batch):
        pn = self.mdl
        def stablize_reward(r):
            r_2D = r.view(-1, self.num_rollouts)
            if self.baseline == 'avg_reward':
                stabled_r_2D = r_2D - r_2D.mean(dim=1, keepdim=True)
            elif self.baseline == 'max_min_scalar':
                stabled_r_2D = (r_2D - torch.min(r_2D,dim=1, keepdim=True)[0])/(torch.max(r_2D,dim=1, keepdim=True)[0] - torch.min(r_2D,dim=1, keepdim=True)[0])
            elif self.baseline == 'avg_reward_normalized':
                stabled_r_2D = (r_2D - r_2D.mean(dim=1, keepdim=True)) / (r_2D.std(dim=1, keepdim=True) + ops.EPSILON)
            else:
                raise ValueError('Unrecognized baseline function: {}'.format(self.baseline))
            stabled_r = stabled_r_2D.view(-1)
            return stabled_r
        #list of (e1,e2,r),size of batchsize
        e1, e2, r = self.format_batch(mini_batch, num_tiles=self.num_rollouts)
        #here become 1 dim temsor,size of batchsize*num_rollouts
        output = self.rollout(e1, r, e2, num_steps=self.num_rollout_steps)

        # Compute policy gradient loss
        pred_e2 = output['pred_e2']
        log_action_probs = output['log_action_probs']
        action_entropy = output['action_entropy']
        path_trace = output["path_trace"]
        #path_trace(steps, (r,e), batch_size)

        baseline_reward = self.reward_fun(e1, r, e2, pred_e2)
        if self.use_conf:
            final_reward = self.conf_fun(
                [(var_to_numpy(r),var_to_numpy(e)) for (r,e) in path_trace],
                var_to_numpy(e1),
                var_to_numpy(r),
                var_to_numpy(e2)
            )
        else:
            final_reward = self.reward_fun(e1, r, e2, pred_e2)

        if self.baseline == 'n/a':
            final_reward = torch.where(final_reward<baseline_reward,final_reward,baseline_reward)
        elif self.baseline == "curriculum":
            final_reward = self.alpha * final_reward + (1-self.alpha) * baseline_reward
        else:
            final_reward = stablize_reward(final_reward)

        if self.plot:
            x = var_to_numpy(baseline_reward)
            y = var_to_numpy(final_reward)
            x = x - 0.5
            y = y - 0.5
            plt.cla()
            plt.scatter(x,y)
            ax = plt.gca()
            ax.spines['right'].set_color('r')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines['bottom'].set_position(('data',0))
            ax.yaxis.set_ticks_position('left')
            ax.spines['left'].set_position(('data',0))
            plt.savefig(os.path.join(self.model_dir,"dist{}.png".format(self.plotid)))
        cum_discounted_rewards = [0] * self.num_rollout_steps
        cum_discounted_rewards[-1] = final_reward
        R = 0
        for i in range(self.num_rollout_steps - 1, -1, -1):
            R = self.gamma * R + cum_discounted_rewards[i]
            cum_discounted_rewards[i] = R

        # Compute policy gradient
        pg_loss, pt_loss = 0, 0
        for i in range(self.num_rollout_steps):
            log_action_prob = log_action_probs[i]
            pg_loss += -cum_discounted_rewards[i] * log_action_prob
            pt_loss += -cum_discounted_rewards[i] * torch.exp(log_action_prob)

        # Entropy regularization
        entropy = torch.cat([x.unsqueeze(1) for x in action_entropy], dim=1).mean(dim=1)
        pg_loss = (pg_loss - entropy * self.beta).mean()
        pt_loss = (pt_loss - entropy * self.beta).mean()

        loss_dict = {}
        loss_dict['model_loss'] = pg_loss
        loss_dict['print_loss'] = float(pt_loss)
        loss_dict['reward'] = final_reward
        loss_dict['entropy'] = float(entropy.mean())
        if self.run_analysis:
            fn = torch.zeros(final_reward.size())
            for i in range(len(final_reward)):
                if not final_reward[i]:
                    if int(pred_e2[i]) in self.kg.all_objects[int(e1[i])][int(r[i])]:
                        fn[i] = 1
            loss_dict['fn'] = fn

        return loss_dict

    def rollout(self, e_s, q, e_t, num_steps, visualize_action_probs=False):
        """
        Perform multi-step rollout from the source entity conditioned on the query relation.
        :param pn: Policy network.
        :param e_s: (Variable:batch) source entity indices.
        :param q: (Variable:batch) query relation indices.
        :param e_t: (Variable:batch) target entity indices.
        :param kg: Knowledge graph environment.
        :param num_steps: Number of rollout steps.
        :param visualize_action_probs: If set, save action probabilities for visualization.
        :return pred_e2: Target entities reached at the end of rollout.
        :return log_path_prob: Log probability of the sampled path.
        :return action_entropy: Entropy regularization term.
        """
        assert (num_steps > 0)
        kg, pn = self.kg, self.mdl

        # Initialization
        log_action_probs = []
        action_entropy = []
        r_s = int_fill_var_cuda(e_s.size(), kg.dummy_start_r)
        seen_nodes = int_fill_var_cuda(e_s.size(), kg.dummy_e).unsqueeze(1)
        path_components = []

        path_trace = [(r_s, e_s)]
        pn.initialize_path((r_s, e_s), kg)

        for t in range(num_steps):
            last_r, e = path_trace[-1]
            obs = [e_s, q, e_t, t==(num_steps-1), last_r, seen_nodes]#起始点，关系，目标点，是否到达步数上限，上一步的关系，已经到过的节点
            db_outcomes, inv_offset, policy_entropy = pn.transit(
                e, obs, kg, use_action_space_bucketing=self.use_action_space_bucketing)
            sample_outcome = self.sample_action(db_outcomes, inv_offset)
            action = sample_outcome['action_sample']
            #here is (r,e) with number format
            pn.update_path(action, kg)
            action_prob = sample_outcome['action_prob']
            log_action_probs.append(ops.safe_log(action_prob))
            action_entropy.append(policy_entropy)
            seen_nodes = torch.cat([seen_nodes, e.unsqueeze(1)], dim=1)
            path_trace.append(action)

            if visualize_action_probs:
                top_k_action = sample_outcome['top_actions']
                top_k_action_prob = sample_outcome['top_action_probs']
                path_components.append((e, top_k_action, top_k_action_prob))

        pred_e2 = path_trace[-1][1]
        self.record_path_trace(path_trace)

        return {
            'pred_e2': pred_e2,
            'log_action_probs': log_action_probs,
            'action_entropy': action_entropy,
            'path_trace': path_trace,
            'path_components': path_components
        }

    def sample_action(self, db_outcomes, inv_offset=None):
        """
        Sample an action based on current policy.
        :param db_outcomes (((r_space, e_space), action_mask), action_dist):
                r_space: (Variable:batch) relation space
                e_space: (Variable:batch) target entity space
                action_mask: (Variable:batch) binary mask indicating padding actions.
                action_dist: (Variable:batch) action distribution of the current step based on set_policy
                    network parameters
        :param inv_offset: Indexes for restoring original order in a batch.
        :return next_action (next_r, next_e): Sampled next action.
        :return action_prob: Probability of the sampled action.
        """

        def apply_action_dropout_mask(action_dist, action_mask):
            if self.action_dropout_rate > 0:
                rand = torch.rand(action_dist.size())
                action_keep_mask = var_cuda(rand > self.action_dropout_rate).float()
                # There is a small chance that that action_keep_mask is accidentally set to zero.
                # When this happen, we take a random sample from the available actions.
                # sample_action_dist = action_dist * (action_keep_mask + ops.EPSILON)
                sample_action_dist = \
                    action_dist * action_keep_mask + ops.EPSILON * (1 - action_keep_mask) * action_mask
                return sample_action_dist
            else:
                return action_dist

        def sample(action_space, action_dist):
            sample_outcome = {}
            ((r_space, e_space), action_mask) = action_space
            sample_action_dist = apply_action_dropout_mask(action_dist, action_mask)
            idx = torch.multinomial(sample_action_dist, 1, replacement=True)
            next_r = ops.batch_lookup(r_space, idx)
            next_e = ops.batch_lookup(e_space, idx)
            action_prob = ops.batch_lookup(action_dist, idx)
            sample_outcome['action_sample'] = (next_r, next_e)
            sample_outcome['action_prob'] = action_prob
            return sample_outcome

        if inv_offset is not None:
            next_r_list = []
            next_e_list = []
            action_dist_list = []
            action_prob_list = []
            for action_space, action_dist in db_outcomes:
                sample_outcome = sample(action_space, action_dist)
                next_r_list.append(sample_outcome['action_sample'][0])
                next_e_list.append(sample_outcome['action_sample'][1])
                action_prob_list.append(sample_outcome['action_prob'])
                action_dist_list.append(action_dist)
            next_r = torch.cat(next_r_list, dim=0)[inv_offset]
            next_e = torch.cat(next_e_list, dim=0)[inv_offset]
            action_sample = (next_r, next_e)
            action_prob = torch.cat(action_prob_list, dim=0)[inv_offset]
            sample_outcome = {}
            sample_outcome['action_sample'] = action_sample
            sample_outcome['action_prob'] = action_prob
        else:
            sample_outcome = sample(db_outcomes[0][0], db_outcomes[0][1])

        return sample_outcome

    def predict(self, mini_batch, verbose=False):
        kg, pn = self.kg, self.mdl
        e1, e2, r = self.format_batch(mini_batch)
        beam_search_output = search.beam_search(
            pn, e1, r, e2, kg, self.num_rollout_steps, self.beam_size)
        pred_e2s = beam_search_output['pred_e2s']
        pred_e2_scores = beam_search_output['pred_e2_scores']
        #batch size 64, beam size 128, len(search_straces) = 3,  search_straces[0] is tuple, search_straces[0][0] size is 8192
        if verbose:
            search_traces = beam_search_output['search_traces']
            output_beam_size = min(self.beam_size, pred_e2_scores.shape[1])
            paths = []
            for i in range(len(e1)):
                # if e1[i] == 0:
                #     continue
                beam = []
                for j in range(output_beam_size):
                    ind = i * output_beam_size + j
                    if pred_e2s[i][j] == kg.dummy_e:
                        break
                    search_trace = []
                    for k in range(len(search_traces)):
                        search_trace.append((int(search_traces[k][0][ind]), int(search_traces[k][1][ind])))
                    beam.append(search_trace)
                paths.append(beam)
        with torch.no_grad():
            pred_scores = zeros_var_cuda([len(e1), kg.num_entities])
            for i in range(len(e1)):
                pred_scores[i][pred_e2s[i]] = torch.exp(pred_e2_scores[i])
                re_paths = None
            if verbose:
                re_paths = [[list() for j in range(kg.num_entities)] for i in range(len(e1))]
                for i in range(len(e1)):
                    for j in range(len(pred_e2s[i])):
                        if pred_e2s[i][j] == kg.dummy_e:
                            break
                        re_paths[i][pred_e2s[i][j]] = paths[i][j]
        return pred_scores,re_paths

    def record_path_trace(self, path_trace):
        path_length = len(path_trace)
        flattened_path_trace = [x for t in path_trace for x in t]
        path_trace_mat = torch.cat(flattened_path_trace).reshape(-1, path_length)
        path_trace_mat = path_trace_mat.data.cpu().numpy()

        for i in range(path_trace_mat.shape[0]):
            path_recorder = self.path_types
            for j in range(path_trace_mat.shape[1]):
                e = path_trace_mat[i, j]
                if not e in path_recorder:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] = 1
                        self.num_path_types += 1
                    else:
                        path_recorder[e] = {}
                else:
                    if j == path_trace_mat.shape[1] - 1:
                        path_recorder[e] += 1
                path_recorder = path_recorder[e]
