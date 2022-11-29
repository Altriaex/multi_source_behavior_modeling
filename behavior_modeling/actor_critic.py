# coding=utf-8


"""Behavior Regularized Actor Critic with estimated behavior policy."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gin
import tensorflow.compat.v1 as tf
from . import agent
from . import networks2
from . import utils
from . import brac_primal_agent


ALPHA_MAX = 500.0
CLIP_EPS = 1e-3

@gin.configurable
class Agent(brac_primal_agent.Agent):
	"""An actor-critic agent using maximum behavior policy set"""

	def __init__(self,**kwargs):
		super(Agent, self).__init__(**kwargs)

	def _init_vars(self, batch):
		self._build_q_loss(batch)
		self._build_p_loss(batch)
		self._q_vars = self._get_q_vars()
		self._p_vars = self._get_p_vars()
		self._load_behavior_policy()
		
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
		self._agent_module.p_net.initialize_policy_vec()
		for qnet, qtarget in self._agent_module.q_nets:
				qnet.initialize_q_vec()
				qtarget.initialize_q_vec()
		
	def _build_fns(self):
		self._agent_module = AgentModule(modules=self._modules)
		self._q_fns = self._agent_module.q_nets
		self._p_fn = self._agent_module.p_net
		self._b_fn = self._agent_module.b_net
		self._qb_fns = self._agent_module.qb_nets


	def _build_q_loss(self, batch):
		s1, s2 = batch['s1'], batch['s2']
		a1 = utils.clip_by_eps(batch['a1'], self._action_spec, 1e-3)
		r, dsc = batch['r'], batch['dsc']
		tid = batch['tid']
		# policy evaluation
		q2_targets = []
		q1_preds = []
		_, a2_p, _ = self._p_fn(s2)
		for q_fn, q_fn_target in self._q_fns:
			q1_preds.append(q_fn(s1, a1))
			q2_targets.append(q_fn_target(s2, a2_p))
		q2_targets = tf.stack(q2_targets, axis=-1)
		q2_target = self.ensemble_q(q2_targets)
		q1_target = tf.stop_gradient(r + dsc * self._discount * q2_target)
		
		q_losses = []
		for ind, q1_pred in enumerate(q1_preds):
			q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
			q_losses.append(q_loss_)
		q_obj_loss = tf.reduce_sum(q_losses)
		# behavior part
		q2b_targets = []
		q1b_preds = []
		_, a2_p, _ = self._p_fn(s2, tid)
		for q_fn, q_fn_target in self._q_fns:
			q1b_preds.append(q_fn(s1, a1, tid))
			q2b_targets.append(q_fn_target(s2, a2_p, tid))
		q2b_targets = tf.stack(q2b_targets, axis=-1)
		q2b_target = self.ensemble_q(q2b_targets)
		q1b_target = tf.stop_gradient(r + dsc * self._discount * q2b_target)
		q_losses = []
		for ind, q1_pred in enumerate(q1b_preds):
			q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1b_target))
			q_losses.append(q_loss_)
		q_bc_loss = tf.reduce_sum(q_losses)
		
		# Ensure variables of behavior policy are loaded
		for q_fn, q_fn_target in self._agent_module._qb_nets:
			_ = q_fn(s1, a1, tid)
			_ = q_fn(s1, a1)
			_ = q_fn_target(s2, a2_p, tid) 
			_ = q_fn_target(s2, a2_p) 
		q_w_norm = self._get_q_weight_norm()
		norm_loss = self._weight_decays[0] * q_w_norm
		loss = q_obj_loss + norm_loss + q_bc_loss
		# info
		info = collections.OrderedDict()
		info['q_obj_loss'] = q_obj_loss
		info['q_bc_loss'] = q_bc_loss
		info['q_norm'] = q_w_norm
		info['r_mean'] = tf.reduce_mean(r)
		info['dsc_mean'] = tf.reduce_mean(dsc)
		info['q2_target_mean'] = tf.reduce_mean(q2_target)
		info['q2b_target_mean'] = tf.reduce_mean(q2b_target)
		return loss, info

	def _build_p_loss(self, batch):
		# read from batch
		s = batch['s1']
		a_b = batch['a1']
		a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
		tid = batch['tid']
		
		_, a_p, log_pi_a_p = self._p_fn(s)
		q1s = []
		for q_fn, _ in self._q_fns:
			q1_ = q_fn(s, a_p)
			q1s.append(q1_)
		q1s = tf.stack(q1s, axis=-1)
		q1 = self._ensemble_q1(q1s)
		p_obj_loss = tf.reduce_mean(- q1)
		
		# The BC part.
		
		log_b_a_b = self._p_fn.get_log_density(s, a_b, tid)
		p_bc_loss = tf.reduce_mean(- log_b_a_b)
		
		# Ensure variables of the behavior policy are loaded
		_ = self._agent_module.b_net.get_log_density(s, a_b, tid)
		_ = self._agent_module.b_net.get_log_density(s, a_b)
		
		p_w_norm = self._get_p_weight_norm()
		norm_loss = self._weight_decays[1] * p_w_norm
		loss = p_obj_loss + norm_loss + p_bc_loss
		# info
		info = collections.OrderedDict()
		info['p_obj_loss'] = p_obj_loss
		info['p_norm'] = p_w_norm
		info['q_a_p_mean'] = tf.reduce_mean(q1)
		return loss, info

	def _build_checkpointer(self):
		state_ckpt = tf.train.Checkpoint(
				policy=self._agent_module.p_net,
				agent=self._agent_module,
				global_step=self._global_step,
				qnets = self._agent_module.q_nets
				)
		behavior_ckpt = tf.train.Checkpoint(
				policy= self._agent_module.b_net,
				qnets = self._agent_module.qb_nets)
		return dict(state=state_ckpt, behavior=behavior_ckpt)

	@tf.function
	def _optimize_step(self, batch):
		info = collections.OrderedDict()
		if tf.equal(self._global_step % self._update_freq, 0):
			source_vars, target_vars = self._get_source_target_vars()
			self._update_target_fns(source_vars, target_vars)
		q_info = self._optimize_q(batch)
		p_info = self._optimize_p(batch)
		info.update(p_info)
		info.update(q_info)
		return info

class AgentModule(brac_primal_agent.AgentModule):
	"""Models in a brac_primal agent."""
	def _build_modules(self):
		n_q_fns = self._modules.n_q_fns
		self._b_net = self._modules.p_net_factory()
		self._qb_nets = []
		for _ in range(n_q_fns):
			self._qb_nets.append(
					[self._modules.q_net_factory(),
					 self._modules.q_net_factory(),]  # source and target
					)
		self._p_net = self._modules.p_net_factory()
		self._q_nets = []
		for _ in range(n_q_fns):
			self._q_nets.append(
					[self._modules.q_net_factory(),
					 self._modules.q_net_factory(),]  # source and target
					)
		self._alpha_var = tf.Variable(1.0)
	@property
	def b_variables(self):
		return self._b_net.trainable_variables

	@property
	def qb_nets(self):
		return self._qb_nets
	
	@property
	def qb_source_variables(self):
		vars_ = []
		for q_net, _ in self._qb_nets:
			vars_ += q_net.trainable_variables
		return tuple(vars_)

	@property
	def qb_target_variables(self):
		vars_ = []
		for _, q_net in self._qb_nets:
			vars_ += q_net.trainable_variables
		return tuple(vars_)

def get_modules(model_params, action_spec, n_trajectory=None):
	"""Get agent modules."""
	model_params, n_q_fns = model_params
	if len(model_params) == 1:
		model_params = tuple([model_params[0]] * 2)
	elif len(model_params) < 2:
		raise ValueError('Bad model parameters %s.' % model_params)
	def q_net_factory():
		return networks2.CriticWithTidNetwork(
			n_trajectory=n_trajectory, fc_layer_params=model_params[0])
	def p_net_factory():
			return networks2.ActorWithTidNetwork(
				action_spec, n_trajectory, fc_layer_params=model_params[1])
	modules = utils.Flags(
		q_net_factory=q_net_factory,
		p_net_factory=p_net_factory,
		n_q_fns=n_q_fns)
	return modules

class Config(agent.Config):
	def _get_modules(self):
		return get_modules(
			self._agent_flags.model_params,
			self._agent_flags.action_spec,
			self._agent_flags.n_trajectory)
