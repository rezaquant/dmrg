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
from typing import Optional, Sequence, Dict, Any, List

import logging
logger = logging.getLogger(__name__)

ar.register_function("torch", "stop_gradient", lambda x: x.detach())
ar.register_function("jax",   "stop_gradient", jax.lax.stop_gradient)
def stop_grad(x):
    return ar.do("stop_gradient", x)



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


def energy_global(MPO_origin, mps_a, opt="auto-hq"):

    mps_a_ = mps_a.copy()
    mps_a_.normalize()
    p_h=mps_a_.H 
    p_h.reindex_(  { f"k{i}":f"b{i}" for i in range(mps_a.L)} )
    MPO_t = MPO_origin *1.0
 
    E_dmrg = (p_h | MPO_t | mps_a_).contract(all,optimize=opt)
    return E_dmrg 



def gate_1d(tn, where, G, ind_id="k{}", site_tags="I{}",
            cutoff=1.e-12, contract='split-gate', 
            inplace=False):

    """
    Apply a 1D gate to a tensor network at one or two sites.

    Args:
        tn:      Tensor network (quimb/qtn TensorNetwork).
        where:   Iterable of site indices; length 1 (single-qubit) or 2 (two-qubit).
        G:       Gate tensor (or matrix).
        ind_id: Format string for site indices (e.g., "k{}" -> "k3").
        site_tags: Format string for site tags   (e.g., "I{}" -> "I3").
        cutoff:  SVD cutoff (used for split contraction paths).
        contract: Contraction mode (e.g., "split-gate") or bool for single-qubit.
        inplace: Modify tn in place if True; otherwise return a new TN.

    Returns:
        TensorNetwork with the gate applied and site tags added.
    """
    
    if len(where)==2:
        x, y = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(x), ind_id.format(y)], contract=contract, inplace=inplace,
                                **{"cutoff":cutoff}
                                    )


        # for s in (x, y):
        #     ind = ind_id.format(s)
        #     tids = tn.ind_map.get(ind)
        #     if tids:
        #         tid = next(iter(tids))
        #         tn.tensor_map[tid].add_tag(site_tags.format(s))

        
        # adding site tags
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(x)] ][0]
        t.add_tag(site_tags.format(x))
        t = [ tn.tensor_map[i] for i in tn.ind_map[ind_id.format(y)] ][0]
        t.add_tag(site_tags.format(y))

    if len(where)==1:
        x, = where
        tn = qtn.tensor_network_gate_inds(tn, G, [ind_id.format(x)], contract=True, inplace=inplace)

    return tn




def internal_inds(psi):
    open_inds = psi.outer_inds()
    innre_inds = []
    for t in psi:
        t_list = list(t.inds)
        for j in t_list :
            if j not in open_inds:
                innre_inds.append(j)
    return innre_inds



def canonize_mps(p, where, cur_orthog):
    xmin, xmax = sorted(where)
    p.canonize([xmin, xmax], cur_orthog=cur_orthog, 
               #info=info_c
              )
    # update cur_orthog in place (preserving reference)
    cur_orthog[:] = [xmin, xmax]


def fidel_mps(psi, psi_fix):

    opt = opt_(progbar=False)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1 ** 2
    f = complex(val_1 / (val_0 * val_) ).real
    return  f




class FIT:
    """
    Fidelity Fitting for tensor networks.

    Parameters
    ----------
    tn : TensorNetwork
        Target tensor network to fit.
    p0 : TensorNetwork, optional
        Initial MPS (starting state). Must support `.copy()` and `.canonize()`.
    cutoffs : float, optional
        Numerical cutoff for truncation (default: 1e-9).
    backend : str or None, optional
        Backend specification for tensor operations.
    n_iter : int, optional
        Number of optimization iterations (default: 4).
    verbose : bool, optional
        If True, logs fidelity at each iteration.
    re_tag : bool, default=True
        If True, (re)tag the target TN for environment construction.
    """

    def __init__(
        self,
        tn: qtn.TensorNetwork,
        p: Optional[qtn.TensorNetwork] = None,
        cutoffs: float = 1.e-12,
        backend: Optional[str] = None,
        site_tag_id: str = "I{}",
        opt: str = "auto-hq",
        range_int: Optional[Sequence[int]] = None,
        re_tag: bool = False,
        info: Optional[Dict[str, Any]] = None,
        warning: bool = False,
        stop_grad_: bool = False, 
    ):

        if p is None:
            raise ValueError("Initial MPS `p` must be provided for FIT.")
        if not isinstance(p, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
            raise TypeError("Initial MPS `p` must be MatrixProductState or MatrixProductOperator.")

        self.L = len(p.tensor_map.keys())

        self.p = p.copy()
        if stop_grad_:
            self.p.apply_to_arrays(stop_grad)

        self.tn = tn.copy()

        
        if site_tag_id:
            self.p.view_as_(qtn.MatrixProductState, L=self.L, 
                            site_tag_id=site_tag_id, site_ind_id=None, 
                            cyclic=False)
        
        
        self.site_tag_id = site_tag_id
        
        
        # contengra path finder
        self.opt = opt

        # cutoffs and underlying backend
        self.cutoffs = cutoffs
        self.backend = backend

        # warnings being pronted or not
        self.warning = warning
        
        # store cost fucntion results
        self.loss: List[float] = []
        self.loss_: List[float] = []
        self.info: Dict[str, Any] = info or {}
        self.range_int: List[int] = list(range_int) if range_int is not None else []
        if self.range_int:
            if len(self.range_int) != 2:
                raise ValueError("range_int must be a sequence of two integers: (start, stop).")
            start, stop = self.range_int
            if start >= stop:
                raise ValueError("range_int must satisfy start < stop.")

        
        # Is there a better solution?
        # Reindex tensor network with random UUIDs for internal indices
        self.tn.reindex_( {idx: qtn.rand_uuid() for idx in self.tn.inner_inds()} )



        
        if set(self.tn.outer_inds()) != set(self.p.outer_inds()):
            raise ValueError("tn and p have different outer indices.")

        
        # re_new tags of tn to be used for effective envs: use interal function "re_tag" to retag
        if re_tag:
            self._re_tag()


    def visual(self, figsize=(14, 14), layout="neato", show_tags=False, tags_: Optional[Sequence[str]] = None, show_inds=False):
        # Visualize network with MPS
        tag_list = tags_ if tags_ is not None else []
        tags = [self.site_tag_id.format(i) for i in range(self.L)] + tag_list
        return (self.tn & self.p).draw(tags, legend=False, show_inds=show_inds,
                                 show_tags=show_tags, figsize=figsize, node_outline_darkness=0.1, 
                                       node_outline_size=None, highlight_inds_color="darkred",
                                      edge_scale=2.0, layout=layout,refine_layout="auto",
                                      highlight_inds=self.p.outer_inds(),
                                      )

    # -------------------------
    # Tagging methods
    # -------------------------
    def _deep_tag(self):
        """
        Propagates tags through the tensor network to ensure every tensor
        receives at least one site tag. Useful for layered TNs.
        """
        tn = self.tn
        count = 1

        while count >= 1:
            tags = tn.tags
            count = 0
            for tag in tags:
                tids = tn.tag_map[tag]
                neighbors = qtn.oset()
                for tid in tids:
                    t = tn.tensor_map[tid]
                    for ix in t.inds:
                        neighbors |= tn.ind_map[ix]
                for tid in neighbors:
                    t = tn.tensor_map[tid]
                    if not t.tags:
                        t.add_tag(tag)
                        count += 1

    def _re_tag(self):
        
        # drop tags
        tn = self.tn
        tn.drop_tags()

        # get outer inds and all tags
        p = self.p
        site_tags = [ self.site_tag_id.format(i) for i in range(p.L)   ]
        inds = list(p.outer_inds())
        

        # smart tagging for the first layer: meaning each tensor in tn is connected directly to p's tensors
        for site_tag in site_tags:
            indx = [i for i in p[site_tag].inds if i in inds][0]
            
            t = [tn.tensor_map[tid] for tid in tn.ind_map[indx]][0]
            
            if not t.tags:
                t.add_tag(site_tag)
                


        if len(tn.tensor_map.keys()) != len(tn.tags):
            if self.warning:
                logger.warning("Missing tags in the tensor network — it’s probably a layered TN.") 
            self._deep_tag()

            
    def run(self, n_iter=6, verbose=True):
        
        """Run the fitting process."""
        if self.p is None:
            raise ValueError("Initial state `p0` must be provided.")

        psi = self.p
        L = self.L        
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        for iteration in range(n_iter):            
            for site in range(L):
                
                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1

                # Canonicalize psi at the current site
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)

                
                psi_h = psi.H.select([site_tag_id.format(site)], "!any")
                tn_ = psi_h | self.tn


                # Contract and normalize
                f = tn_.contract(all, optimize=opt)
                f = f.transpose(*psi[site].inds)

                norm_f = (f.H & f).contract(all) ** 0.5
                self.loss_.append( complex(norm_f).real )
                
                # Update tensor data
                psi[site].modify(data=f.data)

            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))

    def _build_env_right(self, psi, env_right):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        # iterate from rightmost to leftmost
        for i in reversed(range(L)):
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block 

                
            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                # tie to previously computed right environment
                t |= env_right[site_tag_id.format(i+1)]
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)




    def _right_range(self, psi, env_right, start, stop):
        """
        Build right environments env_right["I{i}"] for i in 0..L-1.
        env_right[i] corresponds to contraction of site i and everything to the right (inclusive).
        """
        L = self.L
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        # iterate from rightmost to leftmost
        # for i in reversed(range(L)):
        for count, i in enumerate(range(stop, start, -1)):
            
            psi_block = psi.H.select([site_tag_id.format(i)], "all")

            # Is there any tensor in tn to be included in env
            if site_tag_id.format(i) in self.tn.tags:
                tn_block = self.tn.select([site_tag_id.format(i)], "all")
                t = psi_block | tn_block
            else:
                t = psi_block 

                
            if i == L - 1:
                env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
            else:
                
                if count==0:
                    indx = psi.bond(stop+1, stop)
                    indx_ = self.tn.bond(stop+1, stop)

                    
                # tie to previously computed right environment
                if env_right[site_tag_id.format(i+1)] is not None:
                    t |= env_right[site_tag_id.format(i+1)]
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)
                else:
                    t = t.reindex( {indx:indx_} ) 
                    env_right[site_tag_id.format(i)] = t.contract(all, optimize=opt)

    def _left_range(self, psi, site, count, env_left):
        """Update left environment incrementally for current site."""

        # get tensor at stie from p
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block 
            
        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            if count - 1 == 0:
                indx = psi.bond(site-1, site)
                indx_ = self.tn.bond(site-1, site)
                t = t.copy()
                t = t.reindex( {indx:indx_} )
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
            else:
                t |= env_left[site_tag_id.format(site-1)]
                env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)



    def _update_env_left(self, psi, site: int, env_left):
        """Update left environment incrementally for current site."""
        
        psi_block = psi.H.select([self.site_tag_id.format(site)], "all")
        opt = self.opt
        site_tag_id = self.site_tag_id
        
        if site_tag_id.format(site) in self.tn.tags:
            tn_block = self.tn.select([self.site_tag_id.format(site)], "all")
            t = psi_block | tn_block
        else:
            t = psi_block 
            
        if site == 0:
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)
        else:
            t |= env_left[site_tag_id.format(site-1)]
            env_left[site_tag_id.format(site)] = t.contract(all, optimize=opt)

    
    def run_eff(self, n_iter=6, verbose=True):

        """Run the eefective fitting process"""
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        opt = self.opt

        #info_c = self.info_c
        #range_int = self.range_int 


        
        env_left = { site_tag_id.format(i):None   for i in range(psi.L)}
        env_right = { site_tag_id.format(i):None   for i in range(psi.L)}

        
        for iteration in range(n_iter):    
            
            for site in range(L):


                # Determine orthogonalization reference
                ortho_arg = "calc" if site == 0 else site - 1
                # Canonicalize psi at the current site

                
                psi.canonize(site, cur_orthog=ortho_arg, bra=None)


                self._build_env_right(psi, env_right) if site == 0 else self._update_env_left(psi, site-1, env_left)
                
                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None
                
                if site == 0:
                    if tn_site:
                        tn =  tn_site | env_right[site_tag_id.format(site+1)]
                    else:
                        tn = env_right[site_tag_id.format(site+1)]
                    
                if site > 0 and site < L-1:
                    if tn_site:
                        tn =  tn_site  |  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
            
                if site == L-1:
                    if tn_site:
                        tn =  tn_site | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_left[site_tag_id.format(site-1)]

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(*psi[site_tag_id.format(site)].inds)
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)
                

                norm_f = (f.H & f).contract(all) ** 0.5
                self.loss_.append( complex(norm_f).real )

                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))



    def run_gate(self, n_iter=6, verbose=True):

        """Run the eefective fitting process"""
        if self.p is None:
            raise ValueError("Initial state `p` must be provided.")

        site_tag_id = self.site_tag_id
        psi = self.p
        L = self.L
        opt = self.opt

        if len(self.range_int) != 2:
            raise ValueError("range_int must be set to (start, stop) before calling run_gate.")
        start, stop = self.range_int 
        
        env_left = { site_tag_id.format(i):None   for i in range(psi.L)}
        env_right = { site_tag_id.format(i):None   for i in range(psi.L)}


        for iteration in range(n_iter):    
            
            for i in range(stop, start, -1):
                psi.right_canonize_site(i, bra=None)

            for count_, site in enumerate(range(start, stop+1)):
 
                
                self._right_range(psi, env_right, start, stop) if count_ == 0 else self._left_range(psi, site-1, count_, env_left)

                
                if self.site_tag_id.format(site) in self.tn.tags:
                    tn_site = self.tn.select([site_tag_id.format(site)], "any")
                else:
                    tn_site = None

                if site == 0:
                    if tn_site:
                        tn =  tn_site | env_right[site_tag_id.format(site+1)]
                    else:
                        tn = env_right[site_tag_id.format(site+1)]


                
                if site > 0 and site < L-1:

                    # Boundary consistency: the left and right indices must match between tn and p
                    if count_ == 0:
                        indx = psi.bond(start-1, start)
                        indx_ = self.tn.bond(start-1, start)
                        tn_site = tn_site.reindex({indx_:indx})
                    if count_ == stop  - start:
                        indx = psi.bond(stop+1, stop)
                        indx_ = self.tn.bond(stop+1, stop)
                        tn_site = tn_site.reindex({indx_:indx})
                        
                    
                    if tn_site:
                        if env_right[site_tag_id.format(site+1)] is not None and env_left[site_tag_id.format(site-1)] is not None:
                            tn =  tn_site  |  env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
                        elif env_left[site_tag_id.format(site-1)] is not None:
                            tn =  tn_site | env_left[site_tag_id.format(site-1)]
                        elif env_right[site_tag_id.format(site+1)] is not None:
                            tn =  tn_site | env_right[site_tag_id.format(site+1)]
                        else:
                            tn =  tn_site 
                         
                    else:
                        tn = env_right[site_tag_id.format(site+1)] | env_left[site_tag_id.format(site-1)]
            
                if site == L-1:
                    if tn_site:
                        tn =  tn_site | env_left[site_tag_id.format(site-1)]
                    else:
                        tn =  env_left[site_tag_id.format(site-1)]

                if isinstance(tn, qtn.TensorNetwork):
                    f = tn.contract(all, optimize=opt).transpose(*psi[site_tag_id.format(site)].inds)
                elif isinstance(tn, qtn.Tensor):
                    f = tn.transpose(*psi[site_tag_id.format(site)].inds)


                norm_f = (f.H & f).contract(all) ** 0.5
                # norm_f = ar.do("norm", f.data)
                
                self.loss_.append( complex(norm_f).real )
                
                # Contract and normalize
                # Update tensor data
                psi[site].modify(data=f.data)

                if site < stop:
                    psi.left_canonize_site(site, bra=None)


            # Compute fidelity if verbose mode is enabled
            if verbose:
                fidelity = fidel_mps(self.tn, psi)
                self.loss.append(ar.do("real", fidelity))
