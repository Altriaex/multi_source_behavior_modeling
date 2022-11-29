# coding=utf-8


"""Behavior cloning, policy set and Q functions."""
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
from . import policies


ALPHA_MAX = 500.0
CLIP_EPS = 1e-3


@gin.configurable
class Agent(agent.Agent):
	"""Behavior cloning agent."""
			 
	def __init__(
			self,
			ensemble_q_lambda=1.0,
			**kwargs):
		self._ensemble_q_lambda = ensemble_q_lambda
		super(Agent, self).__init__(**kwargs)

	def _build_fns(self):
		self._agent_module = AgentModule(modules=self._modules)
		self._p_fn = self._agent_module.p_net
		self._q_fns = self._agent_module.q_nets
		self._get_log_density = self._agent_module.p_net.get_log_density

	
	def _get_q_vars(self):
		return self._agent_module.q_source_variables

	def _get_p_vars(self):
		return self._agent_module.p_variables

	def _get_q_weight_norm(self):
		weights = self._agent_module.q_source_weights
		norms = []
		for w in weights:
			norm = tf.reduce_sum(tf.square(w))
			norms.append(norm)
		return tf.add_n(norms)

	def _get_p_weight_norm(self):
		weights = self._agent_module.p_weights
		norms = []
		for w in weights:
			norm = tf.reduce_sum(tf.square(w))
			norms.append(norm)
		return tf.add_n(norms)

	def ensemble_q(self, qs):
		lambda_ = self._ensemble_q_lambda
		return (lambda_ * tf.reduce_min(qs, axis=-1)
						+ (1 - lambda_) * tf.reduce_max(qs, axis=-1))
	
	def _build_q_loss(self, batch):
		s1 = batch['s1']
		s2 = batch['s2']
		a1 = batch['a1']
		r = batch['r']
		dsc = batch['dsc']
		tid = batch['tid']
		
		a1 = utils.clip_by_eps(a1, self._action_spec, CLIP_EPS)
		_, a2_p, _ = self._p_fn(s2, tid)
		
		q2_targets = []
		q1_preds = []
		for q_fn, q_fn_target in self._q_fns:
			q1_preds.append(q_fn(s1, a1, tid))
			q2_targets.append(q_fn_target(s2, a2_p, tid))
		q2_target = self.ensemble_q(tf.stack(q2_targets, axis=-1))
		q1_target = tf.stop_gradient(r + dsc * self._discount * q2_target)
		q_losses = []
		for ind, q1_pred in enumerate(q1_preds):
			q_loss_ = tf.reduce_mean(tf.square(q1_pred - q1_target))
			q_losses.append(q_loss_)
		q_loss = tf.reduce_sum(q_losses)

		q_w_norm = self._get_q_weight_norm()
		norm_loss = self._weight_decays[0] * q_w_norm
		loss = q_loss + norm_loss
		# info
		info = collections.OrderedDict()
		info['q_loss'] = q_loss
		info['q_norm'] = q_w_norm
		info['r_mean'] = tf.reduce_mean(r)
		info['q2_target_mean'] = tf.reduce_mean(q2_target)
		return loss, info

	def _build_p_loss(self, batch):
		s = batch['s1']
		a_b = batch['a1']
		tid = batch['tid']
		a_b = utils.clip_by_eps(a_b, self._action_spec, CLIP_EPS)
		log_pi_a_b = self._get_log_density(s, a_b, tid)
		p_loss = tf.reduce_mean( - log_pi_a_b)
		p_w_norm = self._get_p_weight_norm()
		norm_loss = self._weight_decays[0] * p_w_norm
		loss = p_loss + norm_loss
		
		# Construct information about current training.
		info = collections.OrderedDict()
		info['p_loss'] = p_loss
		info['p_norm'] = p_w_norm
		return loss, info

	def _get_source_target_vars(self):
		return (self._agent_module.q_source_variables,
						self._agent_module.q_target_variables)

	def _build_optimizers(self):
		opts = self._optimizers
		if len(opts) == 1:
			opts = tuple([opts[0]] * 2)
		elif len(opts) < 3:
			raise ValueError('Bad optimizers %s.' % opts)
		self._q_optimizer = utils.get_optimizer(opts[0][0])(lr=opts[0][1])
		self._p_optimizer = utils.get_optimizer(opts[1][0])(lr=opts[1][1])
		if len(self._weight_decays) == 1:
			self._weight_decays = tuple([self._weight_decays[0]] * 2)

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

	def _optimize_q(self, batch):
		vars_ = self._q_vars
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(vars_)
			loss, info = self._build_q_loss(batch)
		grads = tape.gradient(loss, vars_)
		grads_and_vars = tuple(zip(grads, vars_))
		self._q_optimizer.apply_gradients(grads_and_vars)
		return info

	def _optimize_p(self, batch):
		vars_ = self._p_vars
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(vars_)
			loss, info = self._build_p_loss(batch)
		grads = tape.gradient(loss, vars_)
		grads_and_vars = tuple(zip(grads, vars_))
		self._p_optimizer.apply_gradients(grads_and_vars)
		return info

	def _build_test_policies(self):
		policy = policies.DeterministicSoftPolicy(
				a_network=self._agent_module.p_net)
		self._test_policies['main'] = policy
		policy = policies.MaxQSoftPolicy(
				a_network=self._agent_module.p_net,
				q_network=self._agent_module.q_nets[0][0],
				)
		self._test_policies['max_q'] = policy

	def _build_online_policy(self):
		return policies.RandomSoftPolicy(
				a_network=self._agent_module.p_net,
				)

	def _init_vars(self, batch):
		self._build_q_loss(batch)
		self._build_p_loss(batch)
		self._q_vars = self._get_q_vars()
		self._p_vars = self._get_p_vars()
		source_vars, target_vars = self._get_source_target_vars()
		utils.soft_variables_update(source_vars, target_vars)

	def _build_checkpointer(self):
		state_ckpt = tf.train.Checkpoint(
				policy=self._agent_module.p_net,
				agent=self._agent_module,
				global_step=self._global_step,
				qnets = self._agent_module.q_nets
				)
		behavior_ckpt = tf.train.Checkpoint(
				policy=self._agent_module.p_net,
				qnets = self._agent_module.q_nets)
		return dict(state=state_ckpt, behavior=behavior_ckpt)

	def save(self, ckpt_name):
		self._checkpointer['state'].write(ckpt_name)
		self._checkpointer['behavior'].write(ckpt_name + '_behavior')

	def restore(self, ckpt_name):
		self._checkpointer['state'].restore(ckpt_name)


class AgentModule(brac_primal_agent.AgentModule):
	"""Models in a brac_primal agent."""

	def _build_modules(self):
		self._q_nets = []
		n_q_fns = self._modules.n_q_fns
		for _ in range(n_q_fns):
			self._q_nets.append(
					[self._modules.q_net_factory(),
                     self._modules.q_net_factory(),])
		self._p_net = self._modules.p_net_factory()


def get_modules(model_params, action_spec, n_trajectory=None):
	"""Get agent modules."""
	model_params, n_q_fns = model_params
	if len(model_params) == 1:
		model_params = tuple([model_params[0]] * 2)
	elif len(model_params) < 2:
		raise ValueError('Bad model parameters %s.' % model_params)
	def q_net_factory():
		return networks2.CriticWithTidNetwork(
			n_trajectory=n_trajectory,
			fc_layer_params=model_params[0])
	def p_net_factory():
			return networks2.ActorWithTidNetwork(
				action_spec, n_trajectory,
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
            self._agent_flags.n_trajectory)
