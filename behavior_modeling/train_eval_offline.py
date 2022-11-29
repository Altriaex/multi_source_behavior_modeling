# coding=utf-8
# Adapted from https://github.com/Farama-Foundation/D4RL-Evaluations/blob/master/brac/behavior_regularized_offline_rl/brac/train_eval_offline.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation in the offline mode."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import h5py

from absl import logging
import os.path as osp

import gin
import gym
import numpy as np
import tensorflow as tf0
import tensorflow.compat.v1 as tf
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from . import dataset
from . import train_eval_utils
from . import utils

from gym.wrappers import time_limit

from tf_agents.environments import tf_py_environment
from tf_agents.environments import gym_wrapper
import d4rl

def get_offline_data(tf_env, data_file=None, load_tid=False):
    gym_env = tf_env.pyenv.envs[0]
    offline_dataset = gym_env.get_dataset(h5path=data_file)
    dataset_size = len(offline_dataset['observations'])
    tf_dataset = dataset.Dataset(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        size=dataset_size,
        has_tid=load_tid)
    observation_dtype = tf_env.observation_spec().dtype
    action_dtype = tf_env.action_spec().dtype

    offline_dataset['terminals'] = np.squeeze(offline_dataset['terminals'])
    offline_dataset['rewards'] = np.squeeze(offline_dataset['rewards'])
    nonterminal_steps, = np.where(
        np.logical_and(
            np.logical_not(offline_dataset['terminals']),
            np.arange(dataset_size) < dataset_size - 1))
    logging.info('Found %d non-terminal steps out of a total of %d steps.' % (
        len(nonterminal_steps), dataset_size))

    s1 = tf.convert_to_tensor(offline_dataset['observations'][nonterminal_steps],
                                dtype=observation_dtype)
    s2 = tf.convert_to_tensor(offline_dataset['observations'][nonterminal_steps + 1],
                                dtype=observation_dtype)
    a1 = tf.convert_to_tensor(offline_dataset['actions'][nonterminal_steps],
                                dtype=action_dtype)
    a2 = tf.convert_to_tensor(offline_dataset['actions'][nonterminal_steps + 1],
                                dtype=action_dtype)
    discount = tf.convert_to_tensor(
        1. - offline_dataset['terminals'][nonterminal_steps + 1],
        dtype=tf.float32)
    reward = tf.convert_to_tensor(offline_dataset['rewards'][nonterminal_steps],
                                    dtype=tf.float32)
    if load_tid:
        traj_info = h5py.File(data_file[:-5] + "-trajectory-info.hdf5", "r")
        tids = traj_info["tids"]
        tids = tf.convert_to_tensor(
            np.squeeze(tids)[nonterminal_steps], dtype=tf.int32)
        tids += 1
        transitions = dataset.Transition_(
        s1, s2, a1, a2, discount, reward, tids)
    else:
        transitions = dataset.Transition(
        s1, s2, a1, a2, discount, reward)

    tf_dataset.add_transitions(transitions)
    return tf_dataset


def env_factory(env_name, seed=0):
    print(env_name)
    gym_env = gym.make(env_name)
    gym_env.seed(seed)
    gym_spec = gym.spec(env_name)
    if gym_spec.max_episode_steps in [0, None]:  # Add TimeLimit wrapper.
        gym_env = time_limit.TimeLimit(gym_env, max_episode_steps=1000)

    tf_env = tf_py_environment.TFPyEnvironment(
        gym_wrapper.GymWrapper(gym_env))
    return tf_env

def find_best_in_bc_vqvae(behavior_ckpt_file, data_file):
    traj_info = h5py.File(data_file[:-5] + "-trajectory-info.hdf5", "r")
    returns = np.squeeze(traj_info["trajectory_returns"])
    traj_info.close()
    bins = [np.quantile(returns, i / 100.) for i in range(100)]
    returns = np.digitize(returns, bins)
    ckpt_dir, _ = os.path.split(behavior_ckpt_file)
    policy_dics = tf.train.load_variable(os.path.join(ckpt_dir, 'agent'), 'policy/dic_matrix/embeddings/.ATTRIBUTES/VARIABLE_VALUE')[1:]
    policy_tids = tf.train.load_variable(os.path.join(ckpt_dir, 'agent'), 'policy/tid_matrix/embeddings/.ATTRIBUTES/VARIABLE_VALUE')[1:]
    policy_dics = policy_dics / np.linalg.norm(policy_dics, axis=1, keepdims=True)
    policy_tids = policy_tids / np.linalg.norm(policy_tids, axis=1, keepdims=True)
    matching_matrix = np.sum(policy_tids[:, None] * policy_dics[None, ], axis=2)
    matching_inds = np.argmax(matching_matrix, axis=1)
    dic_returns = defaultdict(list)
    for ind, return_val in enumerate(returns):
        dic_returns[matching_inds[ind]].append(return_val)
    means = [np.mean(dic_returns[ind]) for ind in range(len(dic_returns))]
    best_ind = np.argmax(means)
    return best_ind + 1

def draw_embeddings(info_file, log_dir, step):
    os.makedirs(osp.join(log_dir, "visualizations"), exist_ok=True)
    ckpt_dir = osp.join(log_dir, 'agent')
    suffix = '/.ATTRIBUTES/VARIABLE_VALUE'
    policy_dics = tf.train.load_variable(
        ckpt_dir, 'policy/dic_matrix/embeddings' + suffix)[1:]
    policy_vec = tf.train.load_variable(
        ckpt_dir, 'policy/policy_vec' + suffix)
    policy_tids = tf.train.load_variable(
        ckpt_dir, 'policy/tid_matrix/embeddings' + suffix)[1:]
    record = h5py.File(osp.join(log_dir,  "visualizations", f"record-{step}.hdf5"), 'w')
    record.create_dataset("policy_dics", data=policy_dics, compression="gzip")
    record.create_dataset("policy_vec", data=policy_vec, compression="gzip")
    record.create_dataset("policy_tids", data=policy_tids, compression="gzip")
    record.close()
    policy_dics /= np.linalg.norm(policy_dics, axis=1, keepdims=True)
    policy_tids /= np.linalg.norm(policy_tids, axis=1, keepdims=True)
    policy_vec /= np.linalg.norm(policy_vec, axis=1, keepdims=True)
    traj_info = h5py.File(info_file, "r")
    returns = np.array(traj_info["trajectory_returns"])
    traj_info.close()
    bins = [np.quantile(returns, i / 100.) for i in range(100)]
    returns = np.digitize(returns, bins)
    pca = PCA(n_components=2)
    X = policy_tids
    X_embedded = pca.fit_transform(X)
    dic_embedded = pca.transform(policy_dics)
    policy_embedded = pca.transform(policy_vec)

    x = X_embedded[:, 0]
    y = X_embedded[:, 1]
    g = returns
    _, ax = plt.subplots(1,1,figsize=(6,6), constrained_layout=True)
    scatter = ax.scatter(x, y, s=10, c=g, cmap='viridis', label='trajectory vectors')
    _ = ax.scatter(dic_embedded[:, 0], dic_embedded[:, 1], marker="*", s=25, c="r", label="dictionary vectors")
    _ = ax.scatter(policy_embedded[:, 0], policy_embedded[:, 1], marker="^", s=25, c="b", label="policy vector")
    ax.legend(loc='upper left')
    _ = plt.colorbar(scatter, ax=ax).set_label("min-max normalized trajectory return", size=15)

    plt.savefig(osp.join(log_dir,  "visualizations", f"embeddings-{step}.png"))
    plt.close()

@gin.configurable
def train_eval_offline(
    # Basic args.
    log_dir,
    data_file,
    agent_module,
    env_name='HalfCheetah-v2',
    n_train=int(1e6),
    shuffle_steps=0,
    seed=0,
    use_seed_for_data=False,
    # Train and eval args.
    total_train_steps=int(1e6),
    summary_freq=100,
    print_freq=1000,
    save_freq=int(2e4),
    eval_freq=5000,
    n_eval_episodes=20,
    # Agent args.
    model_params=(((200, 200),), 2),
    behavior_ckpt_file=None,
    value_penalty=True,
    alpha=1.0,
    optimizers=(('adam', 0.001),),
    batch_size=256,
    weight_decays=(0,),
    update_freq=1,
    update_rate=0.005,
    discount=0.99,
    n_div_samples=10,
    train_alpha=False,
    warm_start=20000,
    load_tid=False,
    behavior_type='bc',
    n_dic=10,
    commit_coe=0.1):
    """Training a policy with a fixed dataset."""
    # Create tf_env to get specs.
    print('[train_eval_offline.py] env_name=', env_name)
    print('[train_eval_offline.py] data_file=', data_file)
    print('[train_eval_offline.py] agent_module=', agent_module)
    print('[train_eval_offline.py] model_params=', model_params)
    print('[train_eval_offline.py] optimizers=', optimizers)
    print('[train_eval_offline.py] bckpt_file=', behavior_ckpt_file)
    print('[train_eval_offline.py] value_penalty=', value_penalty)
    print('[train_eval_offline.py] train_alpha=', train_alpha)
    if use_seed_for_data:
        tf.set_random_seed(0)
        np.random.seed(seed)
        rand = np.random.RandomState(seed)
        tf_env = env_factory(env_name, seed=seed)
    else:
        tf.set_random_seed(0)
        np.random.seed(0)
        rand = np.random.RandomState(0)
        tf_env = env_factory(env_name, seed=0)
    observation_spec = tf_env.observation_spec()
    action_spec = tf_env.action_spec()

    # Prepare data.
    full_data = get_offline_data(tf_env, data_file, load_tid=load_tid)

    # Split data.
    n_train = min(n_train, full_data.size)
    logging.info('n_train %s.', n_train)


    shuffled_indices = utils.shuffle_indices_with_steps(
        n=full_data.size, steps=shuffle_steps, rand=rand)
    train_indices = shuffled_indices[:n_train]
    train_data = full_data.create_view(train_indices)

    # Create agent.
    agent_flags = utils.Flags(
        observation_spec=observation_spec,
        action_spec=action_spec,
        model_params=model_params,
        optimizers=optimizers,
        batch_size=batch_size,
        weight_decays=weight_decays,
        update_freq=update_freq,
        update_rate=update_rate,
        discount=discount,
        train_data=train_data)
    if load_tid:
        traj_info = h5py.File(data_file[:-5] + "-trajectory-info.hdf5", "r")
        # traj_id starts from 1, but tids starts from 0
        n_trajectory = np.max(traj_info["tids"]) + 1
        setattr(agent_flags, 'n_trajectory', n_trajectory)
        traj_info.close()
    else:
        setattr(agent_flags, 'n_trajectory', None)
    setattr(agent_flags, 'behavior_type', behavior_type)
    if behavior_type in ["bc_vqvae", "bc_mixture"]:
        setattr(agent_flags, 'n_dic', n_dic)
    agent_args = agent_module.Config(agent_flags).agent_args
    my_agent_arg_dict = {}

    for k in vars(agent_args):
        my_agent_arg_dict[k] = vars(agent_args)[k]
    if 'brac_primal' in agent_module.__name__:
        my_agent_arg_dict['behavior_ckpt_file'] = behavior_ckpt_file
        my_agent_arg_dict['value_penalty'] = value_penalty
        my_agent_arg_dict['alpha'] = alpha
        my_agent_arg_dict['n_div_samples'] = n_div_samples
        my_agent_arg_dict['train_alpha'] = train_alpha
        my_agent_arg_dict['target_divergence'] = 0.05
        my_agent_arg_dict['warm_start'] = warm_start
    if "bc_vqvae" in agent_module.__name__:
        my_agent_arg_dict['commit_coe'] = commit_coe
    if "brac_vqvae" in agent_module.__name__:
        my_agent_arg_dict['commit_coe'] = commit_coe
        my_agent_arg_dict['behavior_ckpt_file'] = behavior_ckpt_file
        my_agent_arg_dict['alpha'] = alpha
        my_agent_arg_dict['n_div_samples'] = n_div_samples
        my_agent_arg_dict['best_ind'] =  find_best_in_bc_vqvae(behavior_ckpt_file, data_file)
    print('agent:', agent_module.__name__)
    print('agent_args:', my_agent_arg_dict)
    agent = agent_module.Agent(**my_agent_arg_dict)
    agent_ckpt_name = os.path.join(log_dir, 'agent')

    # Restore agent from checkpoint if there exists one.
    if tf.io.gfile.exists('{}.index'.format(agent_ckpt_name)):
        logging.info('Checkpoint found at %s.', agent_ckpt_name)
        agent.restore(agent_ckpt_name)

    # Train agent.
    train_summary_dir = os.path.join(log_dir, 'train')
    eval_summary_dir = os.path.join(log_dir, 'eval')
    train_summary_writer = tf0.compat.v2.summary.create_file_writer(
        train_summary_dir)
    eval_summary_writers = collections.OrderedDict()
    for policy_key in agent.test_policies.keys():
        eval_summary_writer = tf0.compat.v2.summary.create_file_writer(
            os.path.join(eval_summary_dir, policy_key))
        eval_summary_writers[policy_key] = eval_summary_writer
    eval_results = []

    time_st_total = time.time()
    time_st = time.time()
    step = agent.global_step
    timed_at_step = step
    while step < total_train_steps:
        agent.train_step()
        step = agent.global_step
        if step % save_freq == 0:
            agent.save(agent_ckpt_name)
            if "bc_vqvae" in agent_module.__name__  or "brac_vqvae" in agent_module.__name__:
                draw_embeddings(data_file[:-5] + "-trajectory-info.hdf5", log_dir, step)
            logging.info('Agent saved at %s.', agent_ckpt_name)
        if step % summary_freq == 0 or step == total_train_steps:
            agent.write_train_summary(train_summary_writer)
        if step % print_freq == 0 or step == total_train_steps:
            agent.print_train_info()
        if step % eval_freq == 0 or step == total_train_steps:
            time_ed = time.time()
            time_cost = time_ed - time_st
            logging.info(
                'Training at %.4g steps/s.', (step - timed_at_step) / time_cost)
            eval_result, eval_infos = train_eval_utils.eval_policies(
                tf_env, agent.test_policies, n_eval_episodes)
            eval_results.append([step] + eval_result)
            with open(os.path.join(log_dir, 'results.txt'), 'a') as logfile:
                logfile.write(str(eval_result)+'\n')
            logging.info('Testing at step %d:', step)
            for policy_key, policy_info in eval_infos.items():
                logging.info(utils.get_summary_str(
                    step=None, info=policy_info, prefix=policy_key+': '))
                utils.write_summary(eval_summary_writers[policy_key], step, policy_info)
            time_st = time.time()
            timed_at_step = step
    agent.save(agent_ckpt_name)
    time_cost = time.time() - time_st_total
    logging.info('Training finished, time cost %.4gs.', time_cost)
    return np.array(eval_results)
