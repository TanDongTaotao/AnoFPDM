"""
Helpers for training.
"""

import os
import torch as th
import torch.distributed as dist


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_available():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            dist.init_process_group(
                backend="nccl" if th.cuda.is_available() else "gloo",
                rank=rank,
                world_size=world_size
            )
            
            if th.cuda.is_available():
                th.cuda.set_device(local_rank)
        else:
            # Single GPU setup
            pass

def dev():
    """
    Get the device to use for training.
    """
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    data = th.load(path, **kwargs)
    return data


# 移除分布式参数同步函数
# def sync_params(params):
#     ...

# 移除查找空闲端口函数
# def _find_free_port():
#     ...