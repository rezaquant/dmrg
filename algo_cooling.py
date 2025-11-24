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








class Trck_boundary:

    def __init__(self, tn, opt="auto-hq", chi=4, cutoffs=1.e-10, 
                 to_backend=None, to_backend_=None, flat=False):


        self.tn = tn
        self.opt = opt
        self.chi = chi
        self.to_backend = to_backend
        self.to_backend_ = to_backend_
        
        if not flat:
            self.tnh = self.tn.H
            self.tnh.add_tag('BRA')
            self.tn.add_tag('KET')
            self.norm = self.tnh & self.tn
        else:
            self.norm = tn
            
        self.Ly = self.tn.Ly
        self.Lx = self.tn.Lx
        
        self.cutoffs = cutoffs

        self.mps_b = self._init_left(site_tag_id="X{}", cut_tag_id="Y{}")
        self.mps_b |= self._init_right(site_tag_id="X{}", cut_tag_id="Y{}")
        #self.mps_b |= self._init_right(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mps_b |= self._init_left(site_tag_id = "Y{}", cut_tag_id = "X{}")

        self.mpo_b = self._init_left_(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mpo_b |= self._init_left_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mpo_b |= self._init_right_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        self.mpo_b |= self._init_right_(site_tag_id = "X{}", cut_tag_id = "Y{}")



    
    def _init_left(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            

            mps = tn if count==0 else  tn | p_b_left[cut_tag_id.format(count-1) + "_l"]
        
            
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
            
            # print(site_tags, numbers)
               
            inds_ = []
            inds_size = {}
            
            for tag in site_tags:
                tn_select = mps.select(tag)
                inds_local = [j for j in tn_select.outer_inds() if j in outer_inds]
                inds_.append(inds_local)
                inds_size |= {j: mps.ind_size(j) for j in inds_local}

            
        
            inds_k = {}
            count_ = 0
            for inds in inds_:
                for indx in inds:
                    inds_k |= {f"k{count_}":indx}  
                    count_ += 1
        
        
            # create the nodes, by default just the scalar 1.0
            tensors = [qtn.Tensor() for _ in range(L_mps)]
            
            for i_ in range(L_mps):
                if i_ < (L_mps-1):
                    # add the physical indices, each of size 2
                    tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                    tensors[i_].add_tag(f"I{i_}")
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
                if i_ == L_mps-1:
                    # add the physical indices, each of size 2
                    tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                    tensors[i_].add_tag(f"I{i_}")
        
                    
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.to_backend_)
            p.randomize(seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
            p.normalize()
            p_b_left[cut_tag_id.format(count) + "_l"] = p    

    
        return p_b_left



    def _init_right(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
    
        p_b_right = {}
        
        # iterate from Ly-1 down to 0 (inclusive)
        for count in range(0, self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(self.Ly - 1 - count), "any").copy()
        
            mps = tn if count==0 else  tn | p_b_right[cut_tag_id.format(count - 1) + "_r"]

            
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            # Build regex to capture the integer inside site_tag_id, e.g. "X(\d+)"
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
        
            # Build index lists and sizes for the selected site tags
            inds_ = []
            inds_size = {}
            
            for tag in site_tags:
                tn_select = mps.select(tag)
                inds_local = [j for j in tn_select.outer_inds() if j in outer_inds]
                inds_.append(inds_local)
                inds_size |= {j: mps.ind_size(j) for j in inds_local}

            
            # flatten inds -> new names k0, k1, ... (keeps order consistent with sorted site_tags)
            inds_k = {}
            count_ = 0
            for inds in inds_:
                for indx in inds:
                    inds_k[f"k{count_}"] = indx
                    count_ += 1
        
            # create tensors (one per outer index)
            tensors = [qtn.Tensor() for _ in range(L_mps)]
        
            for i_ in range(L_mps):
                # create physical index k{i_} with recorded size
                tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]])
                tensors[i_].add_tag(f"I{i_}")
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (L_mps - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
        
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L=L_mps, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.to_backend_)
            p.randomize(seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
        
            # store the block at the current cut tag
            p.normalize()
            p_b_right[cut_tag_id.format(count) + "_r"] = p
    
    
        return p_b_right

    
    def _init_left_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        
    
        mps_b = {}
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(count), "any")
                    
            mps = tn if count==0 else  tn | mps_b[cut_tag_id.format(count - 1) + "_l"]

            inds_ = { i:[]  for i in range(self.Lx) }
            for i in range(self.Lx):
                tag = site_tag_id.format(i)
                inds_[i] += [(ix, mps.ind_size(ix)) for ix in mps.select(tag, "any").outer_inds()
                                   if ix in mps.outer_inds()]


            tensors = [qtn.Tensor() for _ in range(self.Lx)]
            for i_ in range(self.Lx):
                for indx, size in inds_[i_]:
                # create physical index k{i_} with recorded size
                    tensors[i_].new_ind(indx, size=size)
                tensors[i_].add_tag(f"I{i_}")
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (self.Lx - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
                
            p = qtn.TensorNetwork(tensors)
            p.view_as_(qtn.MatrixProductState, L=self.Lx, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            
            p.apply_to_arrays(self.to_backend_)
            p.randomize(seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
        
            # store the block at the current cut tag
            p.normalize()

            
            
            mps_b[cut_tag_id.format(count) + "_l"] = p
        return mps_b

    
    
    def _init_right_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any")
                    
            mps = tn if count==0 else  tn | mps_b[cut_tag_id.format(count - 1) + "_r"]
        
        
            inds_ = { i:[]  for i in range(self.Lx) }
            for i in range(self.Lx):
                tag = site_tag_id.format(i)
                inds_[i] += [(ix, mps.ind_size(ix)) for ix in mps.select(tag, "any").outer_inds()
                                   if ix in mps.outer_inds()]

            tensors = [qtn.Tensor() for _ in range(self.Lx)]
            for i_ in range(self.Lx):
                for indx, size in inds_[i_]:
                # create physical index k{i_} with recorded size
                    tensors[i_].new_ind(indx, size=size)
                tensors[i_].add_tag(f"I{i_}")
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (self.Lx - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)

            p = qtn.TensorNetwork(tensors)
            p.view_as_(qtn.MatrixProductState, L=self.Lx, site_tag_id="I{}", site_ind_id=None, cyclic=False)
            
            p.apply_to_arrays(self.to_backend_)
            p.randomize(seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
        
            # store the block at the current cut tag
            p.normalize()

            
            mps_b[cut_tag_id.format(count) + "_r"] = p
        return mps_b
