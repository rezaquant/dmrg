import quimb as qu
import numpy as np
import quimb.tensor as qtn
import os, sys
import itertools
import functools
import torch
from tqdm import tqdm
from time import sleep
import cotengra as ctg
import jax
import jax.numpy as jnp
import autoray as ar
import time
import numpy as np
import re

import logging
logger = logging.getLogger(__name__)


def backend_torch(device = "cpu", dtype = torch.float64, requires_grad=False):
    
    def to_backend(x, device=device, dtype=dtype, requires_grad=requires_grad):
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
    
    return to_backend

def backend_numpy(dtype=np.float64):
    
    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)
    
    return to_backend


def backend_jax(dtype=jnp.float64, device=jax.devices("cpu")[0]):
    # device = jax.devices("cpu")[0]
    # dtype=jnp.float64
    # def to_backend(x, device=device, dtype=dtype):
    #     return jax.device_put(jnp.array(x, dtype=dtype), device)

    
    def to_backend(x, dtype=dtype, device=device):
        arr = jax.device_put(jnp.array(x, dtype=dtype), device)
        return arr

    return to_backend


def opt_(progbar=True, max_repeats=2**9, optlib="cmaes", max_time="rate:1e8",
         alpha=64, target_size=2**34, subtree_size=12):


    # high quality: max_time="equil:128",max_repeats=2**10, optlib=nevergrad
    # terminate search if contraction is cheap: "rate:1e9"

    opt = ctg.ReusableHyperOptimizer(
        minimize=f'combo-{alpha}',
        slicing_opts={'target_size': 2**40},         # first do basic slicing
        slicing_reconf_opts={'target_size': target_size},  # then advanced slicing with reconfiguring
        reconf_opts={'subtree_size': subtree_size},            # then finally just higher quality reconfiguring
        max_repeats=max_repeats,
        parallel=True,  # optimize in parallel
        optlib=optlib,  # an efficient parallel meta-optimizer
        hash_method="b",  # most generous cache hits
        directory="cash/",  # cache paths to disk
        progbar=progbar,  # show live progress
        max_time =max_time,
    )
    return opt


