from circuits import trotter_2D_square_inhomogeneous as circ
import numpy as np
from matplotlib import pyplot as plt
from qiskit import transpile
import quimb as qu
import algo_cooling as algo
import register_ as reg
import quimb.tensor as qtn
from tqdm import tqdm 
import quf
import time
from floquet.circuits import (trotter_2D_square, circ_gates, Z1_ti, Z2_ti, mpo_z2_, simulate, transpile)
import math
import algo_dmrg
import torch
to_backend = algo.backend_torch(device = "cuda", dtype = torch.complex128)
to_backend_ = algo.backend_numpy( dtype = "complex64")

opt = algo.opt_(progbar=False)


chi = 2048
Lx, Ly = 6, 6  # Lx < Ly
L = Lx * Ly

steps = 22
gate_info, depth_map, circ_info = quf.circ_info(Lx=Lx, Ly=Ly, steps=steps)


z2_exact = qu.load_from_disk(f"z2_exact/z2_ext_Lx{Lx}Ly{Ly}")
# z2_exact



mpoz2 = mpo_z2_(circ_info, "Z", where_=L//2, form="left")
mpoz2.apply_to_arrays(to_backend)



p = qtn.MPS_computational_state([0]*L)
p.expand_bond_dimension(chi)
info_c = {}
p.canonicalize_([L//2], cur_orthog='calc', info = info_c)
cur_orthog = [L//2, L//2] 

p.apply_to_arrays(to_backend)


F_l, Fidel_l, Norm_l, norm, tqg_, f = ([], [1], [1], 1, 0, 1)


e_x_ = {}
gamma = [] 
res = {f"F_L{L}_chi{chi}":[], f"Scale_L{L}_chi{chi}":[], f"Z2_L{L}_chi{chi}":[], f"Z2_L{L}":z2_exact}
res |= {f"F_L{L}_chi{chi}":[], f"Norm_L{L}_chi{chi}":[], f"Gamma_L{L}_chi{chi}":[]}

print("chi", chi, Lx, Ly, L)


with tqdm(total=len(gate_info),  desc="dmrg:", leave=True, position=0, 
        colour='CYAN', disable = not True) as pbar:

    for (where, count), G in gate_info.items():

        if len(where) == 1:
            algo_dmrg.gate_1d(p, where, to_backend(G), inplace=True)

        if len(where) == 2:
            tqg_ += 1
            # Canonize around where using cur_orthog
            algo_dmrg.canonize_mps(p, where, cur_orthog)

            # build target tensor-network
            p_g = algo_dmrg.gate_1d(p, where, to_backend(G), contract='split-gate', inplace=False)

            # run dmrg
            fit = algo_dmrg.FIT(p_g, p=p, opt=opt, re_tag=False, range_int=cur_orthog)
            fit.run_gate(n_iter=6, verbose=False)
            #fit.run_eff(n_iter=15, verbose=False)
            p = fit.p

            # get un-normalized fidelity
            # norm = p.norm()
            Fidel_l.append( complex(fit.loss_[-1] ** 2).real )


        
        pbar.set_postfix({"tqg_":tqg_, "F": Fidel_l[-1]})
        pbar.refresh()
        pbar.update(1)

        # measure an observable
        if count+1 in list(depth_map.keys()):
            depth = depth_map[count+1]
            # print("depth", depth, "count", count+1)
            e_x = algo_dmrg.energy_global(mpoz2, p, opt=opt)
            e_x = complex(e_x).real
            e_x_[depth] = complex(e_x).real

            res[f"F_L{L}_chi{chi}"].append( Fidel_l[-1] )
            res[f"Scale_L{L}_chi{chi}"].append( z2_exact[depth]/ e_x )
            res[f"Norm_L{L}_chi{chi}"].append( complex(p.norm()).real )
            res[f"Z2_L{L}_chi{chi}"].append( e_x )
            res[f"Gamma_L{L}_chi{chi}"].append( np.log(z2_exact[depth]/ e_x) / (np.log(Fidel_l[-1])+1.e-12) )




res_dmrg = qu.load_from_disk("store/res_dmrg")
res_dmrg.setdefault(f"L_{L}", [])

print(f"chis for {L}:",  res_dmrg[f"L_{L}"] )
res_dmrg |= res

# print(res_dmrg.keys())
res_dmrg[f"L_{L}"].append(chi)
res_dmrg[f"L_{L}"] = list(set(res_dmrg[f"L_{L}"]))

print(f"chis for {L}:",  res_dmrg[f"L_{L}"] )
qu.save_to_disk(res_dmrg, "store/res_dmrg")
