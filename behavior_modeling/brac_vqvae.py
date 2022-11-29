# coding=utf-8


"""Behavior Regularized Actor Critic with estimated behavior policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf
from . import divergences
from . import utils
from . import actor_critic
from . import bc_vqvae

ALPHA_MAX = 500.0
CLIP_EPS = 1e-3

@gin.configurable
class Agent(actor_critic.Agent):
  """An agent that utilize vq-vae behavior policy."""

  def __init__(self, commit_coe=0.1, best_ind=None, **kwargs):
    self.commit_coe = commit_coe
    self.best_ind = best_ind
    super(Agent, self).__init__(divergence_name="kl2", **kwargs)

  def _div_estimate(self, s, tid):
    return self._divergence.primal_estimate(
        s, self._p_fn, self._p_fn, self._n_div_samples,
        action_spec=self._action_spec, tid=tid)

  def _build_fns(self):
    super(Agent, self)._build_fns()
    self._divergence = divergences.get_divergence(
        name=self._divergence_name)
    self._agent_module.assign_alpha(self._alpha)
    
  def _build_q_loss(self, batch):
    s1, s2 = batch['s1'], batch['s2']
    a1 = utils.clip_by_eps(batch['a1'], self._action_spec, CLIP_EPS)
    r, dsc = batch['r'], batch['dsc']
    tid = batch['tid']
    # policy evaluation
    q2_targets = []
    q1_preds = []
    _, a2_p, _ = self._p_fn(s2)
    for q_fn, q_fn_target in self._q_fns:
      q1_preds.append(q_fn(s1, a1))
      q2_targets.append(q_fn_target(s2, a2_p))
    q2_target = self.ensemble_q(tf.stack(q2_targets, axis=-1))
    div_estimate = self._div_estimate(s2, tid=tid)
    q2_target -= self._get_alpha() * div_estimate
    q1_target = tf.stop_gradient(r + dsc * self._discount * q2_target)
    q_losses = []
    for ind, q1_pred in enumerate(q1_preds):
      q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
      q_losses.append(q_loss_)
    q_obj_loss = tf.reduce_sum(q_losses)
    
    # behavior
    q2b_targets = []
    q1b_preds = []
    _, a2_p, _ = self._p_fn(s2, tid)
    _, _, matched = self._agent_module.p_net.lookup(tid)
    for q_fn, q_fn_target in self._q_fns:
      q1b_preds.append(q_fn(s1, a1, matched))
      q2b_targets.append(q_fn_target(s2, a2_p, matched))
    q2b_target = self.ensemble_q(tf.stack(q2b_targets, axis=-1))
    q1b_target = tf.stop_gradient(r + dsc * self._discount * q2b_target)
    q_losses = []
    for _, q1_pred in enumerate(q1b_preds):
      q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1b_target))
      q_losses.append(q_loss_)
    q_bc_loss = tf.reduce_sum(q_losses)
    loss = q_obj_loss + q_bc_loss
    # info
    info = collections.OrderedDict()
    info['q_obj_loss'] = q_obj_loss
    info['q_bc_loss'] = q_bc_loss
    r_mean = tf.reduce_mean(r)
    info['r_mean'] = r_mean
    dsc_mean = tf.reduce_mean(dsc)
    info['dsc_mean'] = dsc_mean
    q2_target_mean = tf.reduce_mean(q2_target)
    info['q2_target_mean'] = q2_target_mean
    info['q2b_target_mean'] = tf.reduce_mean(q2b_target)
    return loss, info

  def _build_p_loss(self, batch):
    # read from batch
    s = batch['s1']
    a_b = batch['a1']
    a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
    tid = batch['tid']
    
    _, a_p, _ = self._p_fn(s)
    q1s = []
    for q_fn, _ in self._q_fns:
      q1_ = q_fn(s, a_p)
      q1s.append(q1_)
    q1s = tf.stack(q1s, axis=-1)
    q1 = self._ensemble_q1(q1s)

    div_estimate = self._div_estimate(s, tid=tid)
    p_obj_loss = tf.reduce_mean(self._get_alpha() * div_estimate - q1)
    
    encoded_s = self._agent_module.p_net.encode_state(s)
    tid_vec, matched_vec, _ = self._agent_module.p_net.lookup(tid)
    
    tid_dist, _ = self._agent_module.p_net.decode(encoded_s, tid_vec)
    matched_dist, _ = self._agent_module.p_net.decode(encoded_s, matched_vec)

    log_tid_a_b = tid_dist.log_prob(a_b)
    log_matched_a_b = matched_dist.log_prob(a_b)
    rec_loss = tf.reduce_mean(- log_tid_a_b - log_matched_a_b)
    commit_loss = self.commit_coe * tf.reduce_mean(
        - tf.reduce_sum(tid_vec * matched_vec, axis=1), axis=0)

    loss = p_obj_loss  + rec_loss + commit_loss
    # info
    info = collections.OrderedDict()
    info['p_obj_loss'] = p_obj_loss
    info['rec_loss'] = rec_loss
    info['commit_loss'] = commit_loss
    q1_mean = tf.reduce_mean(q1)
    info['q_a_p_mean'] = q1_mean
    div_estimate_mean = tf.reduce_mean(div_estimate)
    info['div_mean'] = div_estimate_mean
    return loss, info

  def load_behavior_variables(self, batch):
    s1 = batch['s1']
    a1 = utils.clip_by_eps(batch['a1'], self._action_spec, CLIP_EPS)
    tid = batch['tid']
    # Ensure variables of behavior policy are loaded
    for q_fn, q_fn_target in self._agent_module._qb_nets:
      _ = q_fn(s1, a1, tid)
      _ = q_fn(s1, a1)
      _ = q_fn_target(s1, a1, tid) 
      _ = q_fn_target(s1, a1) 
    # Ensure variables of the behavior policy are loaded
    _ = self._agent_module.b_net.get_log_density(s1, a1, tid)
    _ = self._agent_module.b_net.get_log_density(s1, a1)  
  
  def _init_vars(self, batch):
    self._build_q_loss(batch)
    self._build_p_loss(batch)
    self.load_behavior_variables(batch)
    self._q_vars = self._get_q_vars()
    self._p_vars = self._get_p_vars()
    self._a_vars = self._agent_module.a_variables
    self._load_behavior_policy()
    # copy parameters of behavior network to its own paramters
    q_source_vars, q_target_vars = self._get_source_target_vars()
    qb_source_vars = self._agent_module.qb_source_variables
    qb_target_vars = self._agent_module.qb_target_variables
    utils.soft_variables_update(
        qb_source_vars,
        q_source_vars)
    utils.soft_variables_update(
        qb_target_vars,
        q_target_vars)
    utils.soft_variables_update(
        self._agent_module._b_net.trainable_variables,
        self._agent_module._p_net.trainable_variables)
    self._agent_module.p_net.initialize_policy_vec(self.best_ind)
    for qnet, qtarget in self._agent_module.q_nets:
        qnet.initialize_q_vec(self.best_ind)
        qtarget.initialize_q_vec(self.best_ind)

class AgentModule(actor_critic.AgentModule):
  pass


class Config(bc_vqvae.Config):
  pass
