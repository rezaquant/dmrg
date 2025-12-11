
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
    """
    Build left/right boundary MPS environments for a PEPS/tn slice and keep 
    to bond dimension `chi`. Accepts either a flattened TN
    (`tn_flat`) or doubled network (`tn_double`) and reuses their X{}/Y{} tag
    structure to initialize boundary legs.

    Parameters:
        tn_flat : TensorNetwork, optional
            Flattened TN slice from which to build boundary MPS.
        opt : str, optional
            Contraction optimization strategy passed to `contract`.
        chi : int, optional
            Maximum bond dimension for boundary MPS.
        tn_double : TensorNetwork, optional
            Doubled TN slice from which to build boundary MPS.
        flat : bool, optional
            Whether to use flattened TN (`tn_flat`) or doubled TN (`tn_double`).
        to_backend : callable, optional
            Function to transfer tensors to desired backend/device.
        seed : int, optional
            Random seed for MPS initialization.
        single_layer : bool, optional
            If True, initializes boundary MPS directly from BRA/KET layers.    
    Warning:
        - Initialization randomizes and compresses boundary MPS in place; don't
          reuse a single instance across concurrent fits.
        - Expects incoming TNs to expose `Ly`/`Lx` and consistent X{}/Y{} tags;
          incorrect tagging or large `chi` can quickly exhaust memory.
        - `to_backend` is applied to all tensors; pass a no-op backend if you
          don't want device transfers during setup.
        - for double layer 'tn_flat = peps' and 'tn_double = peps.H | peps' and 'flat flag' make sense here how to organize the MPS structure. 
        - for single layer 'tn_flat = peps' and 'tn_double = peps' and 'flat = False' makes sense.
    
    returns:    
        mps_b : dict
            Dictionary with the normalized left/right boundary MPS.    
    """

    def __init__(self, tn_flat=None, tn_double=False, opt="auto-hq", chi=8, flat=False,
                 to_backend=None, seed=1, single_layer=False ):

        self.seed = seed
        self.opt = opt
        self.chi = chi
        self.to_backend = to_backend
        self.backend_numpy = backend_numpy(dtype="complex128")
        self.flat = flat

        if not flat:
            self.norm = tn_double.copy()
        else:
            self.norm = tn_flat.copy()
            
        self.Ly = tn_flat.Ly
        self.Lx = tn_flat.Lx
        
        if single_layer:
            self.mps_b = self._init_left_sl(site_tag_id="X{}", cut_tag_id="Y{}")
            self.mps_b |= self._init_right_sl(site_tag_id="X{}", cut_tag_id="Y{}")
        else:
            self.mps_b = self._init_left(site_tag_id="X{}", cut_tag_id="Y{}")
            self.mps_b |= self._init_right(site_tag_id="X{}", cut_tag_id="Y{}")

    
    def _init_left_sl(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        Ly = self.Ly

        for count in range(Ly-1):
            tn = self.norm.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            if self.flat and count==0:
                tn.view_as_(qtn.MatrixProductState, L = len(tn.outer_inds()), site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
                tn.apply_to_arrays(self.to_backend)
                p_b_left[cut_tag_id.format(count) + "_l"] = tn
                continue

            mps = tn if count==0 else  tn | p_b_left[cut_tag_id.format(count-1) + "_l"]
        
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
                tensors[i_].add_tag(site_tag_id.format(i_))
                tensors[i_].new_ind(f'k{i_}', size=inds_size[inds_k[f"k{i_}"]]) 
                if i_ < (L_mps-1):
                    # add the physical indices, each of size 2                                   
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)

 
        
            # construct MPS                
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L = L_mps, site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
            p.apply_to_arrays(self.backend_numpy)
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
            p.normalize()
            p_b_left[cut_tag_id.format(count) + "_l"] = p    

    
        return p_b_left



    def _init_right_sl(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
    
        p_b_right = {}
        
        # iterate from Ly-1 down to 0 (inclusive)
        for count in range(0, self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(self.Ly - 1 - count), "any").copy()
            tn = tn.copy()
            if self.flat and count==0:
                tn.view_as_(qtn.MatrixProductState, L=len(tn.outer_inds()), 
                            site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
                tn.apply_to_arrays(self.to_backend)
                p_b_right[cut_tag_id.format(count) + "_r"] = tn
                continue
        
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
                tensors[i_].add_tag(site_tag_id.format(i_))
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (L_mps - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
        

            # construct MPS
            p = qtn.TensorNetwork(tensors)
            p.reindex_(inds_k)
            p.view_as_(qtn.MatrixProductState, L=L_mps, site_tag_id=site_tag_id, site_ind_id=None, 
                       cyclic=False)
            p.apply_to_arrays(self.backend_numpy)
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
        
            # store the block at the current cut tag
            p.normalize()
            p_b_right[cut_tag_id.format(count) + "_r"] = p
    
    
        return p_b_right

    
    def _init_left(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_left = {}
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(count), "any")
            tn = tn.copy()
            
            if self.flat and count==0:
                tn.view_as_(qtn.MatrixProductState, L=len(tn.outer_inds()), 
                            site_tag_id=site_tag_id, site_ind_id=None, cyclic=False)
                tn.apply_to_arrays(self.to_backend)
                p_b_left[cut_tag_id.format(count) + "_l"] = tn
                continue


            mps = tn if count==0 else  tn | p_b_left[cut_tag_id.format(count - 1) + "_l"]
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
                tensors[i_].add_tag(site_tag_id.format(i_))
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (self.Lx - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)
                

            # construct MPS
            p = qtn.TensorNetwork(tensors)
            p.view_as_(qtn.MatrixProductState, L=self.Lx, site_tag_id=site_tag_id, 
                       site_ind_id=None, cyclic=False)
            
            p.apply_to_arrays(self.backend_numpy)
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)

            p.normalize()

            
            
            p_b_left[cut_tag_id.format(count) + "_l"] = p
        return p_b_left

    
    
    def _init_right(self, site_tag_id = "X{}", cut_tag_id = "Y{}"):
        
        p_b_right = {}
        
        
        for count in range(self.Ly-1):
            tn = self.norm.select(cut_tag_id.format(self.Ly - 1 - count), "any")
            tn = tn.copy()
            if self.flat and count==0:
                tn.view_as_(qtn.MatrixProductState, L=len(tn.outer_inds()), site_tag_id=site_tag_id, 
                            site_ind_id=None, cyclic=False)
                tn.apply_to_arrays(self.to_backend)
                p_b_right[cut_tag_id.format(count) + "_r"] = tn
                continue
                    
            mps = tn if count==0 else  tn | p_b_right[cut_tag_id.format(count - 1) + "_r"]
        
        
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
                tensors[i_].add_tag(site_tag_id.format(i_))
                # create left-to-right bonds: bond i_ <-> i_+1 (same as before)
                if i_ < (self.Lx - 1):
                    tensors[i_].new_bond(tensors[i_ + 1], size=self.chi)


            # construct MPS
            p = qtn.TensorNetwork(tensors)
            p.view_as_(qtn.MatrixProductState, L=self.Lx, site_tag_id=site_tag_id, 
                       site_ind_id=None, cyclic=False)
            
            p.apply_to_arrays(self.backend_numpy)
            p.randomize(seed=self.seed, inplace=True)
            p.apply_to_arrays(self.to_backend)
            p.compress("left", max_bond=self.chi)
        
            p.normalize()

            p_b_right[cut_tag_id.format(count) + "_r"] = p
        return p_b_right
    


class CompBdy:
    """
    Class-based interface for computing boundary MPS environments via DMRG fitting.

    Parameters
    ----------
    norm : TensorNetwork
        Target TN to compress with boundary MPS.
    mpo_b : dict
        Dictionary containing left/right boundary MPS to be optimized.
    Ly : int
        Vertical size of the TN slice.
    opt : str, optional
        Contraction optimization strategy passed to ``contract``.
    cut_tag_id : str, optional
        Tag pattern identifying horizontal cuts in the TN.
    site_tag_id : str, optional
        Tag pattern identifying sites in the TN.
    eq_norms : int or None, optional
        If int, equalizes MPS norms to this value after each fit. If None, skips norm equalization.
    n_iter : int, optional
        Number of DMRG iterations per fit.
    pbar : bool, optional
        If True, displays a progress bar during fitting.
    stop_grad_ : bool, optional
        If True, applies stop gradient to MPSs after fitting.
    fidel_ : bool, optional
        If True, computes and displays fidelity after each fit.
    flat : bool, optional
        If True, indicates that the TN is in flattened form.
    visual_ : bool, optional
        If True, visualizes the fitting TN after each fit.
    re_tag : bool, optional
        If True, re-tags the fitted MPS tensors after each fit.
    max_seperation : int, optional
        Maximum separation between left and right boundary MPS before final contraction.
    re_update : bool, optional
        If True, updates the mpo_b dictionary with the newly fitted MPS after each fit and drop grad.
    """

    def __init__(
        self,
        norm,
        mpo_b,
        Ly,
        opt="auto-hq",
        cut_tag_id="Y{}",
        site_tag_id="X{}",
        eq_norms=False,
        n_iter=4,
        pbar=False,
        stop_grad_=False,
        fidel_=False,
        flat=False,
        visual_=False,
        re_tag=False,
        max_seperation=0,
        re_update=False,
    ):
        self.norm = norm
        self.mpo_b = mpo_b
        self.Ly = Ly
        self.opt = opt
        self.cut_tag_id = cut_tag_id
        self.site_tag_id = site_tag_id
        self.eq_norms = eq_norms
        self.n_iter = n_iter
        self.pbar = pbar
        self.stop_grad_ = stop_grad_
        self.fidel_ = fidel_
        self.flat = flat
        self.visual_ = visual_
        self.re_tag = re_tag
        self.max_seperation = max_seperation
        self.re_update = re_update

        self._update_separation()

    def _update_separation(self):
        if self.max_seperation == 0:
            self.y_left = self.Ly // 2
            self.y_right = self.Ly - (self.Ly // 2)
        elif self.max_seperation == 1:
            self.y_left = (self.Ly // 2) - 1
            self.y_right = self.Ly - (self.Ly // 2)
        else:
            raise ValueError("max_seperation must be 0 or 1.")

    def _fit_one_side(self, side, steps, progress_bar):
        fidel = 1.0
        previous = None

        for count in range(steps):
            if side == "right":
                cut_idx = self.Ly - count - 1
                mpo_key = self.cut_tag_id.format(count) + "_r"
            else:
                cut_idx = count
                mpo_key = self.cut_tag_id.format(count) + "_l"

            tn_slice = self.norm.select(self.cut_tag_id.format(cut_idx), "any")

            if count == 0 and self.flat:
                previous = tn_slice
                continue

            if count == 0:
                tn = tn_slice
            else:
                if previous is None:
                    raise ValueError("Missing previous boundary MPS during fitting.")
                tn = tn_slice | previous

            p = self.mpo_b[mpo_key].copy()
            if previous is not None and count > 0:
                p.exponent = complex(previous.exponent).real

            fit = FIT(
                tn,
                p=p,
                inplace=True,
                site_tag_id=self.site_tag_id,
                opt=self.opt,
                re_tag=self.re_tag,
                stop_grad_=self.stop_grad_,
            )

            if self.visual_:
                fit.visual(figsize=(8, 8), show_inds="bond-size", tags_=[])

            fit.run_eff(n_iter=self.n_iter, verbose=False)

            if self.eq_norms:
                fit.p.equalize_norms_(value=self.eq_norms)

            if self.pbar and self.fidel_:
                fidel = fidel_mps(tn, fit.p)

            if progress_bar is not None:
                progress_bar.set_postfix({"F": complex(fidel).real})
                progress_bar.refresh()
                progress_bar.update(1)

            previous = fit.p

            if self.re_update:
                updated = fit.p.copy()
                updated.apply_to_arrays(stop_grad)
                updated.exponent = 0
                updated.normalize()
                self.mpo_b[mpo_key] = updated

        return previous

    def run(
        self,
        *,
        re_update=None,
        max_seperation=None,
        mpo_b=None,
        norm=None,
        re_tag=None,
        visual_=None,
        flat=None,
        fidel_=None,
        stop_grad_=None,
        pbar=None,
        n_iter=None,
        eq_norms=None,
    ):
        """
        Run the boundary fitting. Any provided keyword overrides the instance defaults for this call.
        """
        if re_update is not None:
            self.re_update = re_update
        if max_seperation is not None:
            self.max_seperation = max_seperation
            self._update_separation()
        if mpo_b is not None:
            self.mpo_b = mpo_b
        if norm is not None:
            self.norm = norm
        if re_tag is not None:
            self.re_tag = re_tag
        if visual_ is not None:
            self.visual_ = visual_
        if flat is not None:
            self.flat = flat
        if fidel_ is not None:
            self.fidel_ = fidel_
        if stop_grad_ is not None:
            self.stop_grad_ = stop_grad_
        if pbar is not None:
            self.pbar = pbar
        if n_iter is not None:
            self.n_iter = n_iter
        if eq_norms is not None:
            self.eq_norms = eq_norms


        with tqdm(
            total=self.y_left + self.y_right,
            desc="bdy_dmrg:",
            leave=True,
            position=0,
            colour="CYAN",
            disable=not self.pbar,
        ) as progress_bar:
            p_previous_r = self._fit_one_side("right", self.y_right, progress_bar)
            p_previous_l = self._fit_one_side("left", self.y_left, progress_bar)

        if p_previous_r is None:
            raise ValueError("Boundary contraction failed: missing left or right boundary MPS.")

        if self.max_seperation == 0:
            tn_f = p_previous_r | p_previous_l
        else:
            if p_previous_l:
                tn_f = p_previous_r | self.norm.select(self.cut_tag_id.format(self.y_left), "any") | p_previous_l
            else:
                tn_f = p_previous_r | self.norm.select(self.cut_tag_id.format(self.y_left), "any")

        main, exp = tn_f.contract(all, optimize=self.opt, strip_exponent=True)
        return main * 10 ** exp




