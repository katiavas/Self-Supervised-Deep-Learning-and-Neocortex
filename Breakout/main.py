import os
import torch.multiprocessing as mp
from parallel_env import ParallelEnv
# import wandb
from memory import Memory

SEED = 111

os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['WANDB_START_METHOD'] = 'thread'
# wandb.init(project='icm', entity="katiavas", dir='./')
if __name__ == '__main__':
    env_id = 'ALE/Breakout-v5'
    mp.set_start_method('spawn', force=True)
    global_ep = mp.Value('i', 0)
    # env_id = 'PongNoFrameskip-v4'

    # env_id = 'CartPole-v1'
    n_threads = 12
    n_actions = 4
    # n_actions = 6

    input_shape = [4, 42, 42]
    ICM = False
    # wandb.run.name = env_id+'/'+str(SEED) + '/ICM='+str(ICM)
    env = ParallelEnv(env_id=env_id, num_threads=n_threads,
                      n_actions=n_actions, global_idx=global_ep,
                      input_shape=input_shape, icm=True)

# CartPole ++> n_actions = 2 , input_shape/input_dims = 4
# Acrobot --> n_actions = 3 , input_shape/input_dims = 6
'''the state-space of the Cart-Pole has four dimensions of continuous values 
and the action-space has one dimension of two discrete values'''
