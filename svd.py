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

import torch
to_backend = algo.backend_torch(device = "cpu", dtype = torch.complex128)
to_backend_ = algo.backend_numpy( dtype = "complex128")

opt = algo.opt_(progbar=False)


Lx, Ly = 14, 4
L = Lx * Ly
steps = 2

#circ = qu.load_from_disk("store/circ")
circ = quf.circ_hydro(Lx=Lx, Ly=Ly, steps = steps, dt = 0.25)
map_1d_2d = { j+Ly*i:(i, j)   for j in range(Ly) for i in range(Lx)}
map_2d_1d = { (i, j):j+Ly*i  for i in range(Lx) for j in range(Ly)}
map_1d = { i+Lx*j:j+Ly*i   for j in range(Ly) for i in range(Lx)}

energy_ = {}
for i in range(Lx):
    x_mpo=qtn.MPO_identity(L, phys_dim=2)*1.e-12
    for j in range(Ly):
        for d in [-1, 1]:
            x_mpo += quf.mpo_z_prod(L, "Z", where_=[ map_2d_1d[(i,j)],map_2d_1d[( (i+d)%Lx ,j)]  ])
            x_mpo += quf.mpo_z_prod(L, "Z", where_=[ map_2d_1d[(i,j)],map_2d_1d[( i ,(j+d)%Ly)]  ])
        x_mpo.compress("left" )

    x_mpo = x_mpo * (1/(4*Ly))
    x_mpo.apply_to_arrays(to_backend)
    x_mpo.compress("left" )
    x_mpo.left_canonize()
    # x_mpo.show()
    energy_[i] = x_mpo

gate_info = algo.gate_info(circ, map_1d)


p = qtn.MPS_computational_state([0]*L)
info_c = {}
p.canonicalize_([L//2], cur_orthog='calc', info=info_c)
p.apply_to_arrays(to_backend)


chi = 128

F_l = []
tqg_ = 0
norm_1 = 1
Fidel_l = [1]
f = 1
f_x_ = {}
e_x_ = {}
depth_map = {504:1, 728:2, 952:3, 1176:4, 1400:5, 1624:6, 1848:7, 2072:8, 2520:10, 2968:12, 3416:14}
norm = 1
with tqdm(total=len(gate_info),  desc="svd:", leave=True, position=0, 
        colour='CYAN', disable = not True) as pbar:

    for (where, count), G in gate_info.items():

        if len(where)==1:
            p.gate_(to_backend(G), where, contract=True, inplace=True)
        else:
            tqg_ += 1
            
            #norm_0 = p.norm()
            p_fix = p.gate(to_backend(G), where, inplace=False, contract=False)
            
            p.gate_nonlocal_(to_backend(G), where, max_bond = chi , info=info_c , method='direct', **{"cutoff":1.e-12})
            #norm = p.norm()


            f = algo.fidel_mps(p, p_fix)
            F_l.append( complex(f).real  )
            #F_l.append( complex(f).real  )
            # measure Fidelity
            Fidel = 0
            for f_ in F_l:
                Fidel += np.log(f_)
            Fidel_l.append( np.exp(Fidel) )

            qu.save_to_disk(Fidel_l, f"store/Fidel_l_svd_{chi}_d{steps}")

 

        pbar.set_postfix({"tqg_":tqg_, "Fidel":Fidel_l[-1], "bnd":p.max_bond() , "Norm": round(complex(norm).real,4)})
        pbar.refresh()
        pbar.update(1)


print("chi", chi, "F_svd", Fidel_l[-1], "norm", complex(norm).real, "steps", steps)

E_x = []
for x, mpo in energy_.items(): 
    e_x = algo.energy_global(mpo, p, opt=opt)
    E_x.append( complex(e_x).real )

qu.save_to_disk(E_x, f"store/e_x_svd_{chi}_d{steps}")
print(E_x)