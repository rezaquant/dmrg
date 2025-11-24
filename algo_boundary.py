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



def opt_(progbar=True):
    opt = ctg.ReusableHyperOptimizer(
        max_repeats=2**8,
        parallel=True,  # optimize in parallel
        optlib="cmaes",  # an efficient parallel meta-optimizer
        hash_method="b",  # most generous cache hits
        directory="cash/",  # cache paths to disk
        progbar=progbar,  # show live progress
    )
    return opt

            


class Trck_boundary:

    def __init__(self, tn, opt="auto-hq", chi=4, cutoffs=1.e-10, to_backend=None, to_backend_=None, Lx=4, Ly=4):


        self.tn = tn
        self.opt = opt
        self.chi = chi
        self.to_backend = to_backend
        self.to_backend_ = to_backend_
        self.Lx = Lx
        self.Ly = Ly
        self.cutoffs = cutoffs

        self.mps_b = self._init_left(site_tag_id = "X{}", cut_tag_id = "Y{}")
        self.mps_b |= self._init_right(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mps_b |= self._init_right(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mps_b |= self._init_left(site_tag_id = "Y{}", cut_tag_id = "X{}")

        self.mpo_b = self._init_left_(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mpo_b |= self._init_left_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mpo_b |= self._init_right_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        self.mpo_b |= self._init_right_(site_tag_id = "X{}", cut_tag_id = "Y{}")



    
    def _init_left(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            
            if count == 0:
                mps = tn
            else:
                mps = tn | p_b_left[cut_tag_id.format(count-1) + "_l"]
        
            
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
                inds_local = []
                for j_ in tn_select.outer_inds():
                    if j_ in outer_inds:
                        inds_local.append(j_)
                        #print( mps.ind_size(j_) )
                        inds_size |= {j_:mps.ind_size(j_)}
                inds_.append(inds_local)
        
        
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
            p.randomize( seed=20, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi, cutoff=self.cutoffs)
            p.normalize()
            p_b_left[cut_tag_id.format(count) + "_l"] = p    
    
    
        return p_b_left



    def _init_right(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
    
        p_b_right = {}
        
        # iterate from Ly-1 down to 0 (inclusive)
        for count in range(0, self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any").copy()
        
            # first iteration (top row) -> initialize; otherwise attach previously built block
            if count == 0:
                mps = tn
            else:
                # when going downward, previously-built block is at count+1
                
                mps = tn | p_b_right[cut_tag_id.format(count - 1) + "_r"]
        
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
                inds_local = []
                for j_ in tn_select.outer_inds():
                    if j_ in outer_inds:
                        inds_local.append(j_)
                        inds_size[j_] = mps.ind_size(j_)
                inds_.append(inds_local)
        
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
            tn = self.tn.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            tn.compress_all(inplace=True, **{"max_bond":chi_1, "canonize_distance": 2, "cutoff":1.e-12})
        
            
            if count == 0:
                mps = tn
            else:
                mps_ = mps_b[cut_tag_id.format(count-1) + "_l"].copy()
                mps_.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
                mps = tn | mps_
                mps.drop_tags(cut_tag_id.format(count-1))
        
        
        
            for i in range(self.Lx): 
                mps.contract_tags_(
                                    site_tag_id.format(i), optimize=self.opt)
            
            mps.fuse_multibonds_()
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
            mps.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
            
            mps.apply_to_arrays(self.to_backend_)
        
            mps.expand_bond_dimension(self.chi, 
                                      rand_strength=0.01
                                     )
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
            mps.normalize()
            mps.apply_to_arrays(self.to_backend)
        
            # mps.draw([f"X{i}" for i in range(Lx)], show_inds="bond-size", show_tags=False)
        
            mps_b[cut_tag_id.format(count) + "_l"] = mps
        return mps_b

    
    
    def _init_right_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.tn.select(cut_tag_id.format(self.Ly - 1 - count), "any")
            tn = tn.copy()
            tn.compress_all(inplace=True, **{"max_bond":chi_1, "canonize_distance": 4, "cutoff":1.e-12})
        
            
            if count == 0:
                mps = tn
            else:
                mps_ = mps_b[cut_tag_id.format(count-1) + "_r"].copy()
                mps_.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
                mps = tn | mps_
                mps.drop_tags(cut_tag_id.format(self.Ly-1-count+1))
        
        
        
            for i in range(self.Lx): 
                mps.contract_tags_(
                                    site_tag_id.format(i), optimize=self.opt)
            
            mps.fuse_multibonds_()
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
        
            mps.compress("left", max_bond=chi_2, cutoff=self.cutoffs)
            
            mps.apply_to_arrays(self.to_backend_)
        
            mps.expand_bond_dimension(self.chi, 
                                      rand_strength=0.01
                                     )
            mps.view_as_(qtn.MatrixProductState, L = self.Lx, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)

            mps.normalize()
            mps.apply_to_arrays(self.to_backend)
        
            # mps.draw([f"X{i}" for i in range(Lx)], show_inds="bond-size", show_tags=False)
        
            mps_b[cut_tag_id.format(count) + "_r"] = mps
        return mps_b
