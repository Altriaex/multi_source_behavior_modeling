# coding=utf-8

import os
import os.path as osp
import shutil

import tensorflow as tf0
from absl import app, flags, logging

gpus = tf0.config.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf0.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf0.config.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)
tf0.compat.v1.enable_v2_behavior()
from behavior_modeling import AGENTS, train_eval_offline

# Flags for offline training.
flags.DEFINE_string('exp_path',
					osp.join(os.getenv('HOME', '/'), 'experiments'),
					'Root directory for experiments')
flags.DEFINE_string('agent_name', 'brac_primal', 'agent name.') 
flags.DEFINE_string('data_dir',
					osp.join(os.getenv('HOME', '/'), 'data', 'd4rl'),
					'Directory for data files')
flags.DEFINE_integer('total_train_steps', int(5e5), '')
flags.DEFINE_integer('bc_train_steps', int(5e5), '')
flags.DEFINE_integer('n_train', int(10e6), '')
flags.DEFINE_integer('save_freq', 5000, '')
flags.DEFINE_float('alpha', 1.0, '')
flags.DEFINE_string('method', 'land', '')
flags.DEFINE_integer('n_dic', 10, '')
flags.DEFINE_float('commit_coe', 0.1, '')
FLAGS = flags.FLAGS

def train_bc():
	path, split = osp.split(FLAGS.exp_path)
	path, env = osp.split(path)
	log_dir = os.path.join(FLAGS.exp_path, "BC")
	os.makedirs(log_dir, exist_ok=True)
	train_eval_offline.train_eval_offline(
		log_dir=log_dir,
		data_file=osp.join(FLAGS.data_dir, env + ".hdf5"),
		agent_module=AGENTS['bc'],
		env_name=env,
		n_train=FLAGS.n_train,
		total_train_steps=FLAGS.bc_train_steps,
		n_eval_episodes=1,
		model_params=((200,200),),
		optimizers=(('adam', 5e-4),),
		seed=int(split),
		use_seed_for_data=True)
	return log_dir

def train_advanced_bc(method, load_tid=True):
	path, split = osp.split(FLAGS.exp_path)
	path, env = osp.split(path)
	datafile = osp.join(FLAGS.data_dir, env + ".hdf5")
	log_dir = os.path.join(FLAGS.exp_path, method)
	os.makedirs(log_dir, exist_ok=True)
	train_eval_offline.train_eval_offline(
		log_dir=log_dir,
		data_file=datafile,
		agent_module=AGENTS[method],
		env_name=env,
		n_train=FLAGS.n_train,
		total_train_steps=FLAGS.bc_train_steps,
		n_eval_episodes=1,
		model_params=(((300, 300, 300), (200, 200, 200),), 2),
		optimizers=(('adam', 1e-4), ('adam', 5e-5),  ('adam', 1e-4)),
		seed=int(split),
		use_seed_for_data=True,
		load_tid=load_tid,
		behavior_type=method,
		update_rate=1e-3,
		eval_freq=5e6,
		n_dic=FLAGS.n_dic,
		commit_coe=FLAGS.commit_coe)
	return log_dir

def main(_):
	os.makedirs(FLAGS.exp_path, exist_ok=True)
	path, split = osp.split(FLAGS.exp_path)
	path, env = osp.split(path)
	exp_base, exp_id = osp.split(path)
	datafile = osp.join(FLAGS.data_dir, env + ".hdf5")
	logging.set_verbosity(logging.INFO)
	if FLAGS.method == "lbrac-v":
		value_penalty = False
		bc_type = "bc_vqvae"
		load_tid = True
		agent_module = 'brac_vqvae'
		bc_log_dir = train_advanced_bc(bc_type, load_tid) 
		opt_params = (('adam', 1e-4), ('adam', 5e-5), ('adam', 1e-5))
		update_rate=1e-3
	elif FLAGS.method == "brac-v":
		value_penalty = True
		bc_log_dir = train_bc()
		bc_type = "bc"
		opt_params = (('adam', 1e-3), ('adam', 3e-5), ('adam', 1e-5))
		agent_module = 'brac_primal'
		update_rate=0.005
	elif FLAGS.method == "brac-p":
		value_penalty = False
		bc_log_dir = train_bc()
		bc_type = "bc"
		opt_params = (('adam', 1e-3), ('adam', 3e-5), ('adam', 1e-5))
		agent_module = 'brac_primal'
		update_rate=0.005
	else:
		raise NotImplementedError
	behavior_ckpt_file = os.path.join(bc_log_dir, 'agent_behavior')
	log_dir = os.path.join(FLAGS.exp_path, FLAGS.method)
	model_arch = (((300, 300, 300), (200, 200, 200),), 2)
	os.makedirs(log_dir, exist_ok=True)
	train_eval_offline.train_eval_offline(
		log_dir=log_dir,
		data_file=datafile,
		agent_module=AGENTS[agent_module],
		env_name=env,
		n_train=FLAGS.n_train,
		total_train_steps=FLAGS.total_train_steps,
		model_params=model_arch,
		optimizers=opt_params,
		behavior_ckpt_file=behavior_ckpt_file,
		value_penalty=value_penalty,
		save_freq=FLAGS.save_freq,
		alpha=FLAGS.alpha,
		seed=int(split),
		use_seed_for_data=True,
		summary_freq=1000,
		load_tid=load_tid,
		behavior_type=bc_type,
		update_rate=update_rate,
		n_dic=FLAGS.n_dic,
		commit_coe=FLAGS.commit_coe)
	archive_name = osp.join(
		exp_base, "agents", "_".join([exp_id, env, split, FLAGS.method]))
	shutil.make_archive(
		base_name=archive_name,
		root_dir=log_dir,
		base_dir=None, format="zip")
	archive_name = osp.join(
		exp_base, "agents", "_".join([exp_id, env, split, bc_type]))
	shutil.make_archive(
		base_name=archive_name, root_dir=bc_log_dir, base_dir=None,
		format="zip")


if __name__ == '__main__':
		app.run(main)
