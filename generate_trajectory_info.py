import os.path as osp
import os
import numpy as np
from absl import logging,  app
import h5py

def generate_d4rl_info(env, h5path):
    print(env)
    dataset = h5py.File(h5path, 'r')
    by_terminals = np.nonzero(dataset["terminals"].astype(np.int32))[0]
    # separate trajectories by terminals first
    end_of_trajectories_ = by_terminals.tolist()
    if len(end_of_trajectories_) == 0:
        # no terminal signal, so all trajectories are separated by timeout
        end_of_trajectories_ = [1000]
    if end_of_trajectories_[-1] != len(dataset["terminals"])-1:
        # add the last index as a end-of-trajectory flag
        end_of_trajectories_.append(len(dataset["terminals"])-1)
    end_of_trajectories = [end_of_trajectories_[0]]
    for i in end_of_trajectories_[1:]:
        while i - end_of_trajectories[-1] > 1000:
            # mujoco tasks use 1000 as the maximum length of trajectories
            end_of_trajectories.append(end_of_trajectories[-1]+1000)
        end_of_trajectories.append(i)
    end_of_trajectories = np.array(end_of_trajectories)
    trajectory_lens = end_of_trajectories[1:] - end_of_trajectories[:-1]
    n_transitions = len(dataset["terminals"])
    logging.info(f"Found {len(end_of_trajectories)} trajectories for {env}.")
    logging.info(f"Avg trajectory length {np.mean(trajectory_lens):.2f}.")
    logging.info(f"# transitions {n_transitions}.")
    tids = np.zeros_like(dataset["terminals"].astype(np.int32))
    returns = []
    start = 0
    for t in range(len(end_of_trajectories)):
        end = end_of_trajectories[t]+1
        tids[start: end] = t
        returns.append(np.sum(dataset["rewards"][start:end]))
        start = end
    if len(tids) > start:
        tids[start:] = t + 1
        returns.append(np.sum(dataset["rewards"][start:]))
    returns = np.array(returns)
    logging.info(f"avg returns {np.mean(returns)}.")
    save_path = h5path[:-5] + "-trajectory-info.hdf5"
    new_dataset = h5py.File(save_path, 'w')
    new_dataset.create_dataset("tids", data=tids, compression="gzip")
    new_dataset.create_dataset(
        "trajectory_returns", data=np.array(returns), compression="gzip")
    new_dataset.close()


def main(argv):
    for env in [ "hopper", "halfcheetah", "walker2d"]:
        for suffix in [ "random" , "medium", "expert", "medium-replay", "medium-expert"]:
            env_full = env + "-" + suffix + "-v2"
            generate_d4rl_info(env_full, osp.join(os.getenv('HOME', '/'), "data", "d4rl", env_full+".hdf5"))

    for env in ["hopper", "halfcheetah", "walker2d"]:
        for n_b in ["1", "2", "3", "4", "5"]:
            env_full = env + "-heterogeneous-" + n_b + "-v1"
            generate_d4rl_info(env_full, 
            osp.join(os.getenv('HOME', '/'), "data", "d4rl", env_full+".hdf5"))
    
if __name__ == '__main__':
    app.run(main)
