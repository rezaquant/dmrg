
import numpy as np
import quimb.tensor as qtn


from algo_dmrg import FIT

import cotengra as ctg
from tqdm import tqdm 

import autoray as ar
import re
import jax

ar.register_function("torch", "stop_gradient", lambda x: x.detach())
ar.register_function("jax",   "stop_gradient", jax.lax.stop_gradient)
def stop_grad(x):
    return ar.do("stop_gradient", x)



def backend_numpy(dtype=np.float64):
    
    def to_backend(x, dtype=dtype):
        return np.array(x, dtype=dtype)
    
    return to_backend

def fidel_mps(psi, psi_fix):

    opt = opt_(progbar=False)
    val_0 = abs((psi.H & psi).contract(all, optimize=opt) )
    val_1 = abs((psi.H & psi_fix).contract(all, optimize=opt))
    val_ = abs((psi_fix.H & psi_fix).contract(all, optimize=opt))

    val_1 = val_1 ** 2
    f = complex(val_1 / (val_0 * val_) ).real
    return  f


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



class BdyMPO:

    def __init__(self, tn, opt="auto-hq", chi=4, norm_tn = False, flat=False,
                 to_backend=None, seed= 1):


        self.tn = tn.copy()
        self.seed = seed
        to_backend_ = backend_numpy(dtype="complex128")

        self.opt = opt
        self.chi = chi
        self.to_backend = to_backend
        self.to_backend_ = to_backend_
        
        if not flat:
            self.norm = norm_tn.copy()
        else:
            self.norm = tn
            
        self.Ly = self.tn.Ly
        self.Lx = self.tn.Lx
        

        self.mps_b = self._init_left(site_tag_id="X{}", cut_tag_id="Y{}")
        self.mps_b |= self._init_right(site_tag_id="X{}", cut_tag_id="Y{}")
        #self.mps_b |= self._init_right(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mps_b |= self._init_left(site_tag_id = "Y{}", cut_tag_id = "X{}")

        self.mps_b |= self._init_left_(site_tag_id = "X{}", cut_tag_id = "Y{}")
        #self.mps_b |= self._init_left_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        #self.mps_b |= self._init_right_(site_tag_id = "Y{}", cut_tag_id = "X{}")
        self.mps_b |= self._init_right_(site_tag_id = "X{}", cut_tag_id = "Y{}")



    
    def _init_left(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            
            mps = tn if count==0 else  tn | p_b_left[cut_tag_id.format(count-1) + "_l_"]
        
            outer_inds = mps.outer_inds()
            L_mps = len(outer_inds)
        
            regex = re.compile("^" + re.escape(site_tag_id).replace("\\{\\}", r"(\d+)") + "$")
            site_tags = [t for t in mps.tags if regex.match(t)]
            numbers = [int(regex.match(t).group(1)) for t in mps.tags if regex.match(t)]
            

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
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
            p.normalize()
            p_b_left[cut_tag_id.format(count) + "_l_"] = p    

    
        return p_b_left



    def _init_right(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
    
        p_b_right = {}
        
        # iterate from Ly-1 down to 0 (inclusive)
        for count in range(0, self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(self.Ly - 1 - count), "any").copy()
            tn = tn.copy()
        
            mps = tn if count==0 else  tn | p_b_right[cut_tag_id.format(count - 1) + "_r_"]

            
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
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
        
            # store the block at the current cut tag
            p.normalize()
            p_b_right[cut_tag_id.format(count) + "_r_"] = p
    
    
        return p_b_right

    
    def _init_left_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        
    
        mps_b = {}
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
                    
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
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
        
            # store the block at the current cut tag
            p.normalize()

            
            
            mps_b[cut_tag_id.format(count) + "_l"] = p
        return mps_b

    
    
    def _init_right_(self, site_tag_id = "X{}", cut_tag_id = "Y{}", chi_1=2, chi_2=2):
        mps_b = {}
        
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(self.Ly - 1 - count), "any")
            tn = tn.copy()
                    
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
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
        
            # store the block at the current cut tag
            p.normalize()

            
            mps_b[cut_tag_id.format(count) + "_r"] = p
        return mps_b




def comp_bdy(norm, mpo_b, Ly, opt="auto-hq", cut_tag_id="Y{}", site_tag_id="X{}" , eq_norms=2, 
             str_r = "_r_", str_l = "_l_", n_iter=10, pbar=False, stop_grad_=True, fidel_=False,
             visual_=False):

    
    y_left = Ly//2
    y_right = Ly - Ly//2
    fidel = 1.

    with tqdm(total=y_left+y_right,  desc="dmrg:", leave=True, position=0, 
            colour='CYAN', disable = not True) as pbar:


        for count in range(y_left):
            
            if count == 0:
                p = mpo_b[cut_tag_id.format(count) + str_r]
                if stop_grad_:
                    p.apply_to_arrays(stop_grad)
                    p.exponent = 0

                tn = norm.select(cut_tag_id.format(Ly-count-1), "any")
            else:
                tn = norm.select(cut_tag_id.format(Ly-count-1), "any") | mpo_b[cut_tag_id.format(count-1)+ str_r]
                p = mpo_b[cut_tag_id.format(count)+ str_r]
                if stop_grad_:
                    p.apply_to_arrays(stop_grad)
                
                if eq_norms:
                    # Carry normalization factor to the next boundary MPO
                    p.exponent = mpo_b[cut_tag_id.format(count-1)+ str_r].exponent


            fit = FIT(tn, p=p, site_tag_id="I{}", opt=opt, re_tag=True, stop_grad_=False)
            
            if visual_:
                fit.visual(figsize=(8,8), show_inds="bond-size", tags_=[])


            #fit.run(n_iter=1, verbose=False)
            fit.run_eff(n_iter=n_iter, verbose=False)
            # print(fit.loss_[-1])

            if eq_norms:
                fit.p.equalize_norms_(value=eq_norms)
            
            if pbar and fidel_:
                fidel = fidel_mps(tn, fit.p)

            pbar.set_postfix({ "F": complex(fidel).real})
            pbar.refresh()
            pbar.update(1)


            mpo_b[cut_tag_id.format(count)+ str_r] = fit.p


        for count in range(y_right):
            
            if count == 0:
                p = mpo_b[cut_tag_id.format(count) + str_l]
                if stop_grad_:
                    p.apply_to_arrays(stop_grad)
                    p.exponent = 0

                tn = norm.select(cut_tag_id.format(count), "any")
            else:
                tn = norm.select(cut_tag_id.format(count), "any") | mpo_b[cut_tag_id.format(count-1)+  str_l]
                p = mpo_b[cut_tag_id.format(count)+ str_l]
                if stop_grad_:
                    p.apply_to_arrays(stop_grad)

                # Carry normalization factor to the next boundary MPO
                if eq_norms:
                    p.exponent = mpo_b[cut_tag_id.format(count-1) + str_l].exponent


            fit = FIT(tn, p=p, site_tag_id="I{}", opt=opt, re_tag=True, stop_grad_=False)
            if visual_:
                fit.visual(figsize=(8,8), show_inds="bond-size", tags_=[])

            #fit.run(n_iter=1, verbose=False)
            fit.run_eff(n_iter=n_iter, verbose=False)
            # print(fit.loss_[-1])

            if eq_norms:
                fit.p.equalize_norms_(value=eq_norms)
            
            if pbar and fidel_:
                fidel = fidel_mps(tn, fit.p)

            pbar.set_postfix({ "F": complex(fidel).real})
            pbar.refresh()
            pbar.update(1)
            

            mpo_b[cut_tag_id.format(count)+ str_l] = fit.p



    tn_right = mpo_b[cut_tag_id.format(y_left-1)+ str_r]
    tn_left = mpo_b[cut_tag_id.format(y_right-1)+ str_l]

    tn_f = tn_right | tn_left
    main, exp = tn_f.contract(all, optimize=opt, strip_exponent=True)


    return  main*10**exp