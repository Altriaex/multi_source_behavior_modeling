# coding=utf-8


"""Behavior cloning using vq-vae to learn a set of policies and their Q functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf
from . import agent
from . import bc_tid_q
from . import networks2
from . import utils
from . import policies


ALPHA_MAX = 500.0
CLIP_EPS = 1e-3


@gin.configurable
class Agent(bc_tid_q.Agent):
	"""Behavior cloning agent."""
			 
	def __init__(
			self,
			commit_coe = 0.1,
			**kwargs):
		self.commit_coe = commit_coe
		super(Agent, self).__init__(**kwargs)
	
	def _build_q_loss(self, batch):
		s1 = batch['s1']
		s2 = batch['s2']
		a1 = batch['a1']
		r = batch['r']
		dsc = batch['dsc']
		tid = batch['tid']
		
		a1 = utils.clip_by_eps(a1, self._action_spec, CLIP_EPS)
		_, a2_p, _ = self._p_fn(s2, tid)
		_, _, matched = self._agent_module.p_net.lookup(tid)
		q2_targets = []
		q1_preds = []
		for q_fn, q_fn_target in self._q_fns:
			q1_preds.append(q_fn(s1, a1, matched))
			q2_targets.append(q_fn_target(s2, a2_p, matched))
		q2_target = self.ensemble_q(tf.stack(q2_targets, axis=-1))
		q1_target = tf.stop_gradient(r + dsc * self._discount * q2_target)
		q_losses = []
		for _, q1_pred in enumerate(q1_preds):
			q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
			q_losses.append(q_loss_)
		q_loss = tf.reduce_sum(q_losses)
		# info
		info = collections.OrderedDict()
		info['q_loss'] = q_loss
		info['r_mean'] = tf.reduce_mean(r)
		info['q2_target_mean'] = tf.reduce_mean(q2_target)
		info['q1_target_mean'] = tf.reduce_mean(q1_target)
		return q_loss, info

	def _build_p_loss(self, batch):
		s = batch['s1']
		a_b = batch['a1']
		tid = batch['tid']
		a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
		
		encoded_s = self._agent_module.p_net.encode_state(s)
		tid_vec, matched_vec, _ = self._agent_module.p_net.lookup(tid)
		tid_dist, _ = self._agent_module.p_net.decode(encoded_s, tid_vec)
		matched_dist, _ = self._agent_module.p_net.decode(encoded_s, matched_vec)

		log_tid_a_b = tid_dist.log_prob(a_b)
		log_matched_a_b = matched_dist.log_prob(a_b)
		rec_loss = tf.reduce_mean(- log_tid_a_b - log_matched_a_b)
		commit_loss = self.commit_coe * tf.reduce_mean(
				- tf.reduce_sum(tid_vec * matched_vec, axis=1), axis=0)
		loss = rec_loss + commit_loss
		
		# Construct information about current training.
		info = collections.OrderedDict()
		info['rec_loss'] = rec_loss
		info['commit_loss'] = commit_loss
		return loss, info

	def _init_vars(self, batch):
		self._build_q_loss(batch)
		self._build_p_loss(batch)
		self._q_vars = self._get_q_vars()
		self._p_vars = self._get_p_vars()
		source_vars, target_vars = self._get_source_target_vars()
		utils.soft_variables_update(source_vars, target_vars)
		
class AgentModule(bc_tid_q.AgentModule):
	"""Models in a brac_primal agent."""
	pass


def get_modules(model_params, action_spec, n_dic=10, n_trajectory=None):
	"""Get agent modules."""
	model_params, n_q_fns = model_params
	
	if len(model_params) == 1:
		model_params = tuple([model_params[0]] * 2)
	elif len(model_params) < 2:
		raise ValueError('Bad model parameters %s.' % model_params)
	def q_net_factory():
		return networks2.CriticWithTidNetwork(
			n_trajectory=n_dic,
			fc_layer_params=model_params[0])
	def p_net_factory():
			return networks2.VQVAEActorNetwork(
				action_spec, n_trajectory, n_dic=n_dic,
				fc_layer_params=model_params[1])
	modules = utils.Flags(
				q_net_factory=q_net_factory, p_net_factory=p_net_factory,
				n_q_fns=n_q_fns)
	return modules


class Config(agent.Config):
	def _get_modules(self):
		return get_modules(
				self._agent_flags.model_params,
				self._agent_flags.action_spec,
				self._agent_flags.n_dic,
				self._agent_flags.n_trajectory)
