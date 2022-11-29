# coding=utf-8
# Copyright 2022 Guoxi Zhang and Hisashi Kashima.
#
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

"""Neural network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from .networks import ActorNetwork, CriticNetwork
tfd = tfp.distributions

class ActorWithTidNetwork(ActorNetwork):
	"""Actor network that utilize trajectory id."""

	def __init__(
			self,
			action_spec,
			n_trajectory,
			fc_layer_params=()
			):
		super(ActorWithTidNetwork, self).__init__(action_spec,fc_layer_params)
		self.norm_layer = tf.keras.layers.LayerNormalization()
		self.tid_matrix = tf.keras.layers.Embedding(
									n_trajectory, 8)
		self.policy_vec = tf.Variable([[0.] * 8], dtype=tf.float32)

	def initialize_policy_vec(self):
		mean_vec = tf.reduce_mean(self.tid_matrix.embeddings, axis=0, keep_dims=True)
		self.policy_vec.assign(mean_vec)

	def __call__(self, state, tid=None):
		a_dist, a_tanh_mode = self._get_outputs(state, tid)
		a_sample = a_dist.sample()
		log_pi_a = a_dist.log_prob(a_sample)
		return a_tanh_mode, a_sample, log_pi_a

	def _get_outputs(self, state, tid=None):
		h = state
		batchsize = tf.shape(state)[0]
		for l in self._layers[:-1]:
			h = l(h)
		h = self.norm_layer(h)
		if tid is None:
			tid_vec = tf.math.l2_normalize(self.policy_vec, axis=1)
			tid_vec = tf.repeat(tid_vec, repeats=batchsize, axis=0)
		else:
			tid_vec = self.tid_matrix(tid)
			tid_vec = tf.math.l2_normalize(tid_vec, axis=1)
		h = tf.concat([h, tid_vec], axis=1)
		for l in self._layers[-1:]:
			h = l(h)
		mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
		a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
		std = tf.exp(log_std)
		a_distribution = self.create_action_distribution(mean, std)
		return a_distribution, a_tanh_mode

	def get_log_density(self, state, action, tid=None):
		a_dist, _ = self._get_outputs(state, tid)
		log_density = a_dist.log_prob(action)
		return log_density

	def sample_n(self, state, n=1, tid=None):
		a_dist, a_tanh_mode = self._get_outputs(state, tid)
		a_sample = a_dist.sample(n)
		a_sample = tf.clip_by_value(
			a_sample, self._action_spec.minimum + 1e-3, self._action_spec.maximum - 1e-3)
		log_pi_a = a_dist.log_prob(a_sample)
		return a_tanh_mode, a_sample, log_pi_a

	def init_vecs(self):
		matrix = self.tid_matrix.embeddings
		first_vec = matrix[0][None, :]
		new_init_vals = tf.repeat(first_vec, repeats=matrix.shape[0], axis=0)
		matrix.assign(new_init_vals)
		self.policy_vec.assign(first_vec)

class VQVAEActorNetwork(ActorNetwork):
	"""Actor network that utilize trajectory id."""

	def __init__(
			self,
			action_spec,
			n_trajectory,
			n_dic,
			fc_layer_params=(),
		):
		super(VQVAEActorNetwork, self).__init__(action_spec, fc_layer_params)
		self.norm_layer = tf.keras.layers.LayerNormalization()
		self.tid_matrix = tf.keras.layers.Embedding(
									n_trajectory+1, 8)
		self.dic_matrix = tf.keras.layers.Embedding(
									n_dic+1, 8)
		self.dic_inds = tf.convert_to_tensor([i for i in range(1, n_dic+1)], dtype=tf.int32)
		self.policy_vec = tf.Variable([[100.] * 8], dtype=tf.float32)
		self.n_dic = n_dic

	def get_log_density(self, state, action):
		a_dist, _ = self._get_outputs(state)
		log_density = a_dist.log_prob(action)
		return log_density

	def compute_entropy(self):
		# dic_vecs: n_dic, 8
		dic_vecs = tf.math.l2_normalize(self.dic_matrix.embeddings[1:], axis=1)
		# n_dic, 1, 8 * 1, n_dic, 8 -> n_dic, n_dic
		dist = tf.reduce_sum(dic_vecs[:, None] * dic_vecs[None], axis=2) + 1
		norm = tf.reduce_sum(dist, axis=1, keep_dims=True) - 2
		prob = dist / norm
		entropies = []
		for i in range(self.n_dic):
				for j in range(self.n_dic):
						if j == i:
								continue
						else:
								entropies.append(-prob[i,j] * tf.log(prob[i,j] + 1e-12))
		entropy = tf.reduce_sum(entropies)
		return entropy
				

	def initialize_policy_vec(self, best_ind=None):
		if best_ind is None:
				mean_vec = tf.reduce_mean(self.dic_matrix.embeddings[1:], axis=0, keep_dims=True)
				self.policy_vec.assign(mean_vec)
		else:
				self.policy_vec.assign(self.dic_matrix.embeddings[best_ind][None])
	
	def encode_state(self, state):
		h = state
		batchsize = tf.shape(state)[0]
		for l in self._layers[:2]:
			h = l(h)
		h = self.norm_layer(h)
		return h

	def lookup(self, tid):
		# tid: batchsize, 
		batchsize = tf.shape(tid)[0]
		# tid_vecs: batchsize, 8
		tid_vecs = tf.math.l2_normalize(self.tid_matrix(tid), axis=1)
		dic_inds = tf.repeat(self.dic_inds[None], repeats=batchsize, axis=0)
		# dic_vecs: batchsize, n_dic, 8
		dic_vecs = tf.math.l2_normalize(self.dic_matrix(dic_inds), axis=2)
		# dist: batchsize, n_dic
		dist = tf.reduce_sum(tid_vecs[:, None] * dic_vecs, axis=2)
		# matched: batchsize,
		matched = tf.argmax(dist, axis=1, output_type=tf.int32)
		# tf.keras.Embedding assumes embedding index starts from 1
		matched = matched + 1
		matched_vecs = tf.math.l2_normalize(self.dic_matrix(matched), axis=1)
		return tid_vecs, matched_vecs, matched   

	def decode(self, state_vec, tid_vec):
		h = tf.concat([state_vec, tid_vec], axis=1)
		for l in self._layers[2:]:
			h = l(h)
		mean, log_std = tf.split(h, num_or_size_splits=2, axis=-1)
		a_tanh_mode = tf.tanh(mean) * self._action_mags + self._action_means
		log_std = tf.clip_by_value(log_std, -10, 10)
		std = tf.exp(log_std)
		a_distribution = self.create_action_distribution(mean, std)
		return a_distribution, a_tanh_mode
		
	def __call__(self, state, tid=None):
		a_dist, a_tanh_mode = self._get_outputs(state, tid)
		a_sample = a_dist.sample()
		a_sample = tf.clip_by_value(
			a_sample, self._action_spec.minimum + 1e-3, self._action_spec.maximum - 1e-3)
		log_pi_a = a_dist.log_prob(a_sample)
		return a_tanh_mode, a_sample, log_pi_a

	def _get_outputs(self, state, tid=None):
		state_vec = self.encode_state(state)
		batchsize = tf.shape(state)[0]
		if tid is None:
			tid_vec = tf.math.l2_normalize(self.policy_vec, axis=1)    
			vecs = tf.repeat(tid_vec, repeats=batchsize, axis=0)
		else:
			_, vecs, _ = self.lookup(tid)
		a_distribution, a_tanh_mode = self.decode(state_vec, vecs)
		return a_distribution, a_tanh_mode

	def get_log_density(self, state, action, tid=None):
		a_dist, _ = self._get_outputs(state, tid)
		log_density = a_dist.log_prob(action)
		return log_density

	def sample_n(self, state, n=1, tid=None):
		a_dist, a_tanh_mode = self._get_outputs(state, tid)
		a_sample = a_dist.sample(n)
		a_sample = tf.clip_by_value(
			a_sample, self._action_spec.minimum + 1e-3, self._action_spec.maximum - 1e-3)
		log_pi_a = a_dist.log_prob(a_sample)
		return a_tanh_mode, a_sample, log_pi_a


class CriticWithTidNetwork(CriticNetwork):
	"""Critic Network using tid."""

	def __init__(
			self,
			n_trajectory,
			fc_layer_params=(),
			):
		super(CriticWithTidNetwork, self).__init__(fc_layer_params)
		self.norm_layer = tf.keras.layers.LayerNormalization()
		self.tid_matrix = tf.keras.layers.Embedding(
									n_trajectory+1, 8)
		self.q_vec = tf.Variable([[100.] * 8], dtype=tf.float32)
		self.n_trajectory = n_trajectory

	def initialize_q_vec(self, best_ind=None):
		if best_ind is None:
				mean_vec = tf.reduce_mean(self.tid_matrix.embeddings[1:], axis=0, keep_dims=True)
				self.q_vec.assign(mean_vec)
		else:
				self.q_vec.assign(self.tid_matrix.embeddings[best_ind][None])

	def __call__(self, state, action, tid=None):
		state = tf.cast(state, dtype=tf.float64)
		batchsize = tf.shape(state)[0]
		action = tf.cast(action, dtype=tf.float64)
		h = tf.concat([state, action], axis=-1)
		for l in self._layers[:2]:
			h = l(h)
		h = self.norm_layer(h)
		if tid is None:
			tid_vec = tf.math.l2_normalize(self.q_vec, axis=1)
			tid_vec = tf.repeat(tid_vec, repeats=batchsize, axis=0)
		else:
			tid_vec = self.tid_matrix(tid)
			tid_vec = tf.math.l2_normalize(tid_vec, axis=1)
		h = tf.concat([h, tid_vec], axis=1)
		for l in self._layers[2:]:
			h = l(h)
		return tf.reshape(h, [-1])

	@property
	def weights(self):
		w_list = []
		for ind, l in enumerate(self._layers):
			w_list.append(l.weights[0])
		return w_list
