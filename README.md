# Behavior Estimation from Multi-Source Data for Offline Reinforcement Learning

This is the official code for our AAAI 2023 paper _Behavior Estimation from Multi-Source Data for Offline Reinforcement Learning_. A preprint version of this paper is available on [arXiv](https://arxiv.org/abs/2211.16078), and a video is available on [YouTube](https://youtu.be/T_yLCgJ5a7Q). The code is developed from the [D4RL_Evaluation repository](https://github.com/Farama-Foundation/D4RL-Evaluations). If you find it useful for your research, please kindly cite our paper using:

```
@inproceedings{behavior_estimate,
	address = {Washington, DC, USA},
	author = {Guoxi Zhang and Hisashi Kashima},
	booktitle = {Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence},
	publisher = {AAAI Press},
	title = {Behavior Estimation from Multi-Source Data for Offline Reinforcement Learning},
	year = {2023}}

```

## Installation
The code uses mujoco210, mujoco-py, gym, and legacy versions of tensorflow. We plan to upgrade it to use latest  gymnasium and tensorflow. For now, please follow the instructions below. We use \<user_name\> for your username in a Linux system.

We recommend using [Miniconda3](https://docs.conda.io/en/latest/miniconda.html) to setup a veritual environment. The recommended way to set up these experiments is via a virtualenv. After installing Miniconda3, create an environment like (replace <env_name> with a name you prefer). Make sure your environment is installed at /home/<user_name>/miniconda3/envs/<env_name>/.

```shell
conda create -n <env_name> python=3.9
conda activate <env_name>
```

Then you probably need to install cudatoolkit and cudnn library. Skip this part if you have already installed them. Here we install cudatoolkit 11.6 and cudnn 8.4.1.

```shell
conda install -c conda-forge cudatoolkit=11.6 cudnn=8.4.1
```

Now install mujoco 2.1.0. You may refer instructions listed [here](https://github.com/openai/mujoco-py). Suppose you install at /home/<user_name>/.mujoco/mujoco210/. Then do:

```shell
export MUJOCO_PY_MUJOCO_PATH=/home/<user_name>/.mujoco/mujoco210/
```

Then install the D4RL package.

```shell
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

To run our code, you will need to additionally install the following.

```shell
pip install h5py matplotlib gin-config tensorflow-probability==0.15.0 tf-agents==0.11.0 protobuf~=3.19.0 gtimer scikit-learn
pip install -U gym==0.23.1
```

## Getting Started
First download d4rl datasets from [here](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/). Below we use **halfcheetah-random-v2** as an example. Note that we replace underscores in names of datasets with dashes. Let's put the file at ~/data/d4rl. Then clone our code to your machine.

```
git clone https://github.com/Altriaex/multi_source_behavior_modeling.git
cd multi_source_behavior_modeling
```
We need to generate an info file for each dataset. You may run the script generate_trajectory_info.py to generate info files for all the datasets. Below we invoke a Python interpreter and generate the info file for **halfcheetah-random-v2**.

```shell
python
```

```python
import os.path as osp
import os
import numpy as np
from absl import logging,  app
import gym
import h5py
from generate_trajectory_info import generate_d4rl_info

env_full = "halfcheetah-random-v2"
generate_d4rl_info(env_full, osp.join(os.getenv('HOME', '/'), "data", "d4rl", env_full+".hdf5"))
```
A file named **halfcheetah-random-v2-trajectory-info.hdf5** should be created at 
~/data/d4rl. 
```
sudo apt-get install python-pip
python -m pip install --user virtualenv
python -m virtualenv ~/env
source ~/env/bin/activate
```


Now create a directory for your experiments, such as ~/experiments. Below we use <exp_dir> for the path of this directory. Use the following command to train a BRAC-v agent on halfcheetah-random-v2 with seed=1. This command uses the gpu0.

Note that if you did not install the cudatoolkit and cudnn using the commands above, you probably need to adjust the export command. You can go with export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin and do what ever the error message suggests.


```
export MUJOCO_PY_MUJOCO_PATH=/home/<username>/.mujoco/mujoco210/;export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin:/home/<username>/miniconda3/envs/<env_name>/lib:/usr/lib/nvidia;CUDA_VISIBLE_DEVICES=0 python train.py --exp_path <exp_dir>/alternatives/halfcheetah-random-v2/1 --method brac-v
```

Use the following to train a LBRAC-v agent which learns ten behavior policies. 

```
export MUJOCO_PY_MUJOCO_PATH=/home/<username>/.mujoco/mujoco210/;export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<username>/.mujoco/mujoco210/bin:/home/<username>/miniconda3/envs/<env_name>/lib:/usr/lib/nvidia;CUDA_VISIBLE_DEVICES=0 python train.py --exp_path <exp_dir>/behavior_learning_10/halfcheetah-random-v2/1 --method brac-v --n_dic 10
```