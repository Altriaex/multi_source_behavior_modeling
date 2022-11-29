# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Divergences for BRAC agents."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
import tensorflow.compat.v1 as tf
from . import utils
EPS = 1e-8  # Epsilon for avoiding numerical issues.
CLIP_EPS = 1e-3  # Epsilon for clipping actions.


@gin.configurable
def gradient_penalty(s, a_p, a_b, c_fn, gamma=5.0):
  """Calculates interpolated gradient penalty."""
  batch_size = s.shape[0]
  alpha = tf.random.uniform([batch_size])
  a_intpl = a_p + alpha[:, None] * (a_b - a_p)
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a_intpl)
    c_intpl = c_fn(s, a_intpl)
  grad = tape.gradient(c_intpl, a_intpl)
  slope = tf.sqrt(EPS + tf.reduce_sum(tf.square(grad), axis=-1))
  grad_penalty = tf.reduce_mean(tf.square(tf.maximum(slope - 1.0, 0.0)))
  return grad_penalty * gamma


class Divergence(object):
  """Basic interface for divergence."""

  def dual_estimate(self, s, a_p, a_b, c_fn):
    raise NotImplementedError

  def dual_critic_loss(self, s, a_p, a_b, c_fn):
    return (- tf.reduce_mean(self.dual_estimate(s, a_p, a_b, c_fn))
            + gradient_penalty(s, a_p, a_b, c_fn))

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    raise NotImplementedError


class FDivergence(Divergence):
  """Interface for f-divergence."""

  def dual_estimate(self, s, a_p, a_b, c_fn):
    logits_p = c_fn(s, a_p)
    logits_b = c_fn(s, a_b)
    return self._dual_estimate_with_logits(logits_p, logits_b)

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    raise NotImplementedError

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    _, apn, apn_logp = p_fn.sample_n(s, n_samples)
    _, abn, abn_logb = b_fn.sample_n(s, n_samples)
    # Clip actions here to avoid numerical issues.
    apn_logb = b_fn.get_log_density(
        s, utils.clip_by_eps(apn, action_spec, CLIP_EPS))
    abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    return self._primal_estimate_with_densities(
        apn_logp, apn_logb, abn_logp, abn_logb)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    raise NotImplementedError


class KL(FDivergence):
  """KL divergence."""

  def _dual_estimate_with_logits(self, logits_p, logits_b):
    return (- utils.soft_relu(logits_b)
            + tf.log(utils.soft_relu(logits_p) + EPS) + 1.0)

  def _primal_estimate_with_densities(
      self, apn_logp, apn_logb, abn_logp, abn_logb):
    return tf.reduce_mean(apn_logp - apn_logb, axis=0)

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None):
    _, apn, apn_logp = p_fn.sample_n(s, n_samples)
    _, abn, abn_logb = b_fn.sample_n(s, n_samples)
    # Clip actions here to avoid numerical issues.
    apn_logb = b_fn.get_log_density(
        s, utils.clip_by_eps(apn, action_spec, CLIP_EPS))
    abn_logp = p_fn.get_log_density(
        s, utils.clip_by_eps(abn, action_spec, CLIP_EPS))
    div = self._primal_estimate_with_densities(
        apn_logp, apn_logb, abn_logp, abn_logb)
    div = tf.nn.relu(div)
    return div

class KL2(KL):
  """KL divergence."""

  def primal_estimate(self, s, p_fn, b_fn, n_samples, action_spec=None, tid=None):
    _, apn, apn_logp = p_fn.sample_n(s, n_samples)
    # Clip actions here to avoid numerical issues.
    apn_logb = b_fn.get_log_density(
        s, utils.clip_by_eps(apn, action_spec, CLIP_EPS), tid)
    div = self._primal_estimate_with_densities(apn_logp, apn_logb, None, None)
    # when the two dist. differs too much, sample-based estimates might becomes negative
    div = tf.nn.relu(div)
    return div


CLS_DICT = dict(
    kl=KL,
    kl2=KL2)

def get_divergence(name):
  return CLS_DICT[name]()
