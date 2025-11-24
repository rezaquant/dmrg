from qiskit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, NoiseModel, pauli_error
import qiskit.quantum_info as qi
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit, fsolve
from qiskit.quantum_info import StabilizerState, Pauli, Clifford, random_clifford, random_pauli, SparsePauliOp, entropy, partial_trace
#from qiskit.compiler import transpile
import pickle
#from multiprocessing import Pool
#from scipy.stats import bootstrap
#from functools import partial
#from leakage_detection import *
#from pytket import Circuit
#from pytket.qasm import circuit_to_qasm_str
#from math import comb
from os.path import isfile
from time import time
from qiskit.quantum_info import schmidt_decomposition as sd

from pytket.extensions.qiskit import qiskit_to_tk
import qiskit.qasm2
#import quimb as qu
#import quimb.tensor as qtn
#import cotengra as ctg
#from tqdm import tqdm
import math
#from . import quf
from qiskit import transpile

def Zavg_shot(shot):
    
    num = len(shot)
    return sum([1-2*int(_) for _ in shot])/num

def Zavg_shots(shots):
    
    return [Zavg_shot(_) for _ in shots]
    
def Z2avg_shot(shot):
    
    num = len(shot)
    return sum([1-2*int(_) for _ in shot])**2/num**2

def Z2avg_shots(shots):
    
    return [Z2avg_shot(_) for _ in shots]

def dict_to_XY(d):
    tups = sorted(d.items())
    x = [tup[0] for tup in tups]
    y = [tup[1] for tup in tups]
    return np.array(x),np.array(y)

def resamp(results):
    l = len(results)
    dat = list(np.random.choice(results,l))
    return dat

def resamp_dict(d):

    return {key:resamp(d[key]) for key in d}

def cost(circ, shots, quiet = False):

    gates = circ.count_ops()
    
    hqc = (gates['rx']+10*gates['rzz']+5*gates['measure'])/5000
    return shots*hqc
    
def moving_average(y,w):
    l = len(y)
    mavg = []
    for j in range(len(y)):
        list = [y[k] for k in range(max(j-w,0),j+1)]
        mavg.append(np.mean(list))
    return mavg



def save(thing, filename, overwrite = False):
    
    if isfile(filename) and not overwrite:
        raise Exception('Filename already exists, if you want to overwrite it set optional argument overwrite = True')
    
    with open(filename, 'wb') as file: 
        pickle.dump(thing, file) 



def load(filename):

    with open(filename, 'rb') as file: 
        thing = pickle.load(file) 
    return thing

def circle_dist(i,j,c):
    
    cwdist = max(i,j)-min(i,j)
    ccwdist = int(np.mod(min(i,j)-max(i,j),c))
    
    return min(cwdist,ccwdist)

def weights(x_dim,y_dim):

    sites = [(i,j) for i in range(x_dim) for j in range(y_dim)]
    groups = {}
    for a in sites:
        for b in sites:
            if b != a:
                rx = circle_dist(a[0],b[0],x_dim)
                ry = circle_dist(a[1],b[1],y_dim)
                dist = (rx,ry)
                if dist not in groups:
                    groups[dist] = set()
                    groups[dist].add((a,b))
                else:
                    groups[dist].add((a,b))
    group_lens = {dist:len(groups[dist]) for dist in groups}      
    
    return group_lens
    
def Z1(x_dim,y_dim):
    n = x_dim*y_dim
    op = SparsePauliOp.from_sparse_list([('Z',[i],1/n) for i in range(n)], num_qubits=n)
    return op

def Z2(x_dim,y_dim):
    n = x_dim*y_dim
    off_diags = [('ZZ',[i,j],1/n**2) for i in range(n) for j in range(n) if i!=j]
    diags = [('I',[0],1/n)]
    op = SparsePauliOp.from_sparse_list(off_diags+diags, num_qubits=n)
    return op

def Z1_ti(x_dim,y_dim):
    n = x_dim*y_dim
    op = SparsePauliOp.from_sparse_list([('Z',[0],1)], num_qubits=n)
    return op


def Z2_ti(x_dim,y_dim):
    
    n = x_dim*y_dim
    ws = weights(x_dim,y_dim)
    sites = np.array(range(y_dim*x_dim)).reshape((y_dim,x_dim))
    off_diags = []
    for dist in ws:
        dx = dist[0]
        dy = dist[1]
        inda = 0
        indb = sites[dy,dx]
        off_diags.append(('ZZ',[inda,indb],ws[dist]/n**2))
    
    diags = [('I',[0],1/n)]
    op = SparsePauliOp.from_sparse_list(off_diags+diags, num_qubits=n)
    return op


def process(data):
    
    ret_data = dict()
    for item in data:
        name, time = str.split(item,'_t')
        if name not in ret_data:
            ret_data[name] = []
        ret_data[name].append((int(time),data[item]))
        
    ret_data_sorted = {key:sorted(ret_data[key]) for key in ret_data}
    
    return ret_data_sorted
    
def trotter_2D_square(x_dim: int, y_dim: int, steps: int, dt: float, J: float, h: float, hz: float, theta: float, obs = None, psi = False, psi_fin = False, meas = False, boundary = 'pbc', order = 'second', output = 'qiskit'):

    if order not in ['first','second']:
        raise Excpetion('Only first and second order trotter expansions have been implemented')
    
    N=x_dim*y_dim
    xd=x_dim
    yd=y_dim
    site=np.array(range(yd*xd)).reshape((yd,xd))
    qc = QuantumCircuit(N)
    
    
    #initial state |theta> = cos(theta)|0> + sin(theta)|1>
    if theta==0:
        None
    else:
        for ii in range(N):
            qc.u(2*theta,0,0,ii)
    # compute expectation values for the initial state:
    # if obs:
    #     for loc in obs:
    #         for op in obs[loc]:
    #             qc.save_expectation_value(Pauli(op), qubits=list(loc), label=f't{0}_loc{loc}_'+op)
    if obs:
        for key in obs:
            qc.save_expectation_value(obs[key], range(N), label = key + f'_t0')
    if psi:
        qc.save_statevector(label = f'psi_t0')
    # Apply trotter layers

    for s in range(steps):

        # Nothing or exp(-i h dt X/2), depending on order
        if order == 'second':
            for ii in range(N):
                qc.rx(2*h*dt/2,ii)
            
        # exp(-i hz dt Z)
        if hz != 0:
            for ii in range(N):
                qc.rz(2*hz*dt,ii)


        # exp(-i J dt ZZ)
        # x-direction interactions
        if (xd % 2)==0:
            for y in range(yd):
                for ii in range(0,xd, 2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=2:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
        else:
            for y in range(yd):
                for ii in range(0,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=1:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
    
        # y-direction interactions
        if (yd % 2)==0:
            for x in range(xd):
                for ii in range(0,yd, 2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=2:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])                
        else:        
            for x in range(xd):
                for ii in range(0,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=1:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])

        # exp(-i h dt X) or exp(-i h dt X/2), dependeing on order
        for ii in range(N):
            if order == 'second':
                qc.rx(2*h*dt/2,ii)
            else:
                qc.rx(2*h*dt,ii)
                
    
        # if obs:
        #         for loc in obs:
        #     for op in obs[loc]:
        #         qc.save_expectation_value(Pauli(op), qubits=list(loc), label=f't{s+1}_j{loc}_'+op)
        if obs:
            for key in obs:
                qc.save_expectation_value(obs[key], range(N), label = key + f'_t{s+1}')
                    
        if psi:
            qc.save_statevector(label = f'psi_t{s+1}')        
    if psi_fin:
        qc.save_statevector(label = 'psi_fin', pershot = True)  
    if meas:
        qc.measure_all()
        
    if output == 'qiskit':            
        return qc
    elif output == 'pytket':
        return qiskit_to_tk(qc)
    else:
        raise Exception('Output type not supported, optional output argument must be set to "qiskit" or "pytket".')


def trotter_2D_square_inhomogeneous(x_dim: int, y_dim: int, steps: int, dt: float, J: float, h: float, hz: float, thetas: np.ndarray, obs = None, psi = False, psi_fin = False, meas = False, boundary = 'pbc', order = 'second', output = 'qiskit'):
    """
    Drew's version of trotter_2D_square modified for inhomogeneous initial state specified by array of thetas
    new inputs:
    - thetas: N = x_dim*y_dim np.array of floats, specifies rotation angle theta for each site
    """

    if order not in ['first','second']:
        raise Excpetion('Only first and second order trotter expansions have been implemented')
    
    N=x_dim*y_dim
    xd=x_dim
    yd=y_dim
    site=np.array(range(yd*xd)).reshape((yd,xd))
    qc = QuantumCircuit(N)
    
    
    #initial state |theta> = cos(theta)|0> + sin(theta)|1>
    #if theta==0:
    #    None
    #else:
    for ii in range(N):
        qc.u(2*thetas[ii],0,0,ii)
    # compute expectation values for the initial state:
    # if obs:
    #     for loc in obs:
    #         for op in obs[loc]:
    #             qc.save_expectation_value(Pauli(op), qubits=list(loc), label=f't{0}_loc{loc}_'+op)
    if obs:
        for key in obs:
            qc.save_expectation_value(obs[key], range(N), label = key + f'_t0')
    if psi:
        qc.save_statevector(label = f'psi_t0')
    # Apply trotter layers

    for s in range(steps):

        # Nothing or exp(-i h dt X/2), dependeing on order
        if order == 'second':
            for ii in range(N):
                qc.rx(2*h*dt/2,ii)
            
        # exp(-i hz dt Z)
        if hz != 0:
            for ii in range(N):
                qc.rz(2*hz*dt,ii)


        # exp(-i J dt ZZ)
        # x-direction interactions
        if (xd % 2)==0:
            for y in range(yd):
                for ii in range(0,xd, 2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=2:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
        else:
            for y in range(yd):
                for ii in range(0,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=1:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
    
        # y-direction interactions
        if (yd % 2)==0:
            for x in range(xd):
                for ii in range(0,yd, 2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=2:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])                
        else:        
            for x in range(xd):
                for ii in range(0,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=1:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])

        # exp(-i h dt X) or exp(-i h dt X/2), dependeing on order
        for ii in range(N):
            if order == 'second':
                qc.rx(2*h*dt/2,ii)
            else:
                qc.rx(2*h*dt,ii)
                
    
        # if obs:
        #         for loc in obs:
        #     for op in obs[loc]:
        #         qc.save_expectation_value(Pauli(op), qubits=list(loc), label=f't{s+1}_j{loc}_'+op)
        if obs:
            for key in obs:
                qc.save_expectation_value(obs[key], range(N), label = key + f'_t{s+1}')
                    
        if psi:
            qc.save_statevector(label = f'psi_t{s+1}')        
    if psi_fin:
        qc.save_statevector(label = 'psi_fin', pershot = True)  
    if meas:
        qc.measure_all()
        
    if output == 'qiskit':            
        return qc
    elif output == 'pytket':
        return qiskit_to_tk(qc)
    else:
        raise Exception('Output type not supported, optional output argument must be set to "qiskit" or "pytket".')


def simulate(qc, timing = False, unitary = True):

    if timing:
        begin = time.time()

    if unitary:
        simulator = simulator=AerSimulator(method='statevector')
        result = simulator.run(qc).result()
        data = result.data()
    else:
        simulator = simulator=AerSimulator(method='statevector')
        result = simulator.run(qc, shots = Nshots).result()
        data = result.data()
        

    if timing:
        end = time.time()
        print(f'Simulation time: {end-begin} seconds')
        
    return process(data)

def emulate(qc, shots, noise_model = None):

    simulator = AerSimulator(noise_model = noise_model)
    result = simulator.run(qc, shots = shots, memory = True).result()
    return result.get_memory(qc)

def emulate_fidelity(qc, shots, noise_model = None, max_shots=1):

    if shots%max_shots != 0:
        raise Exception('"max_shots" argument must evenly divide "shots" argument')
        
    simulator = AerSimulator(noise_model = None)
    result = simulator.run(qc, shots = 1).result()
    psi0 = result.data()['psi_fin'][0]
    
    simulator = AerSimulator(noise_model = noise_model)

    all_fids = []
    all_samps = []
    rounds = shots//max_shots
    
    for _ in range(rounds):
        
        result = simulator.run(qc, shots = max_shots, memory = True).result()
        psis = result.data()['psi_fin']
        fids = [np.abs(np.dot(np.conjugate(psi0),psi))**2 for psi in psis]
        samps = result.get_memory(qc)
        
        all_fids = all_fids + fids
        all_samps = all_samps + samps
        
    return all_fids, all_samps 

def vn_entropy_slower(psi, qubits = 'automatic', timing = False):

    if timing:
        begin = time()
        
    if qubits == 'automatic':
        qubits = range(len(psi.dims())//2)
        
    rho = partial_trace(psi,qubits)
    ent = entropy(rho)
    
    if timing:
        end = time()
        print(end-begin)
        
    return ent

def vn_entropy(psi, qubits = 'automatic', timing = False):

    if timing:
        begin = time()
        
    if qubits == 'automatic':
        qubits = list(range(len(psi.dims())//2))
        
    svd = sd(psi,qubits)
    lambdas = np.array([_[0]**2 for _ in svd])
    vne = -np.dot(lambdas,np.log2(lambdas))

    if timing:
        end = time()
        print(end-begin)
            
    return vne


def circuit_1st(x_dim: int, y_dim: int, steps: int, dt: float, J: float, h: float, hz: float, theta: float, geometry='square', obs = {(0):'Z'}, boundary='pbc'):

    N=x_dim*y_dim
    xd=x_dim
    yd=y_dim

    site=np.array(range(yd*xd)).reshape((yd,xd))
    
    qc = QuantumCircuit(N)
    
    #initial state |theta> = cos(theta)|0> + sin(theta)|1>
    if theta==0:
        None
    else:
        for ii in range(N):
            qc.u(2*theta,0,0,ii)

    for j in obs:
        for op in obs[j]:
            qc.save_expectation_value(Pauli(op), qubits=list(j), label=op+f'_t{0}_j{j}')
        
    for s in range(steps):
        
        # exp(-i hz dt Z)
        for ii in range(N):
            qc.rz(2*hz*dt,ii)


        # exp(-i J dt ZZ)
        # x-direction interactions
        if (xd % 2)==0:
            for y in range(yd):
                for ii in range(0,xd, 2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=2:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
        else:
            for y in range(yd):
                for ii in range(0,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=1:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
    


        # y-direction interactions
        if (yd % 2)==0:
            for x in range(xd):
                for ii in range(0,yd, 2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=2:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])                
        else:        
            for x in range(xd):
                for ii in range(0,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=1:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])


        # diagonal interaction for triangular lattice (only for PBC)
        if geometry=='triangular':
            if xd==2 and yd==2:
                qc.rzz(2*J*dt, site[0,0], site[1,1])
                qc.rzz(2*J*dt, site[0,1], site[1,0])
            elif xd==1 or yd==1:
                None
            else:
                for y in range(yd):
                    for x in range(xd):
                        qc.rzz(2*J*dt, site[y,x], site[y-1,(x+1)%xd])


        # exp(-i h dt X)
        for ii in range(N):
            qc.rx(2*h*dt,ii)
    
        for j in obs:
            for op in obs[j]:
                qc.save_expectation_value(Pauli(op), qubits=list(j), label=op+f'_t{s+1}_j{j}')       
    
    return qc

def circuit_2nd(x_dim: int, y_dim: int, steps: int, dt: float, J: float, h: float, hz: float, theta: float, geometry='square', boundary='pbc'):

    N=x_dim*y_dim
    xd=x_dim
    yd=y_dim

    site=np.array(range(yd*xd)).reshape((yd,xd))
    
    qc = QuantumCircuit(N)
    
    #initial state |theta> = cos(theta)|0> + sin(theta)|1>
    if theta==0:
        None
    else:
        for ii in range(N):
            qc.u(2*theta,0,0,ii)


    for s in range(steps):
        
        # exp(-i h dt/2 X)
        for ii in range(N):
            qc.rx(2*h*dt/2,ii)


        # exp(-i J dt ZZ)
        # x-direction interactions
        if (xd % 2)==0:
            for y in range(yd):
                for ii in range(0,xd, 2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=2:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
        else:
            for y in range(yd):
                for ii in range(0,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                for ii in range(1,xd-1,2):
                    qc.rzz(2*J*dt,site[y,ii],site[y,ii+1])
                if boundary=='pbc':
                    if xd!=1:
                        qc.rzz(2*J*dt,site[y,0],site[y,xd-1])
    


        # y-direction interactions
        if (yd % 2)==0:
            for x in range(xd):
                for ii in range(0,yd, 2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=2:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])                
        else:        
            for x in range(xd):
                for ii in range(0,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                for ii in range(1,yd-1,2):
                    qc.rzz(2*J*dt,site[ii,x],site[ii+1,x])
                if boundary=='pbc':
                    if yd!=1:
                        qc.rzz(2*J*dt,site[0,x],site[yd-1,x])


        # diagonal interaction for triangular lattice (only for PBC)
        if geometry=='triangular':
            if xd==2 and yd==2:
                qc.rzz(2*J*dt, site[0,0], site[1,1])
                qc.rzz(2*J*dt, site[0,1], site[1,0])
            elif xd==1 or yd==1:
                None
            else:
                for y in range(yd):
                    for x in range(xd):
                        qc.rzz(2*J*dt, site[y,x], site[y-1,(x+1)%xd])


        # exp(-i h dt/2 X)
        for ii in range(N):
            qc.rx(2*h*dt/2,ii)
            
    return qc



'''-------------------tDMRG functions------------------------'''

def exact_obs(obs, where, circ_info, steps=4):

    to_backend, opt_, opt = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)

    
    x_dim = circ_info["x_dim"]
    y_dim = circ_info["y_dim"]
    theta = circ_info["theta"]
    J = circ_info["J"]
    h = circ_info["h"]
    hz = circ_info["hz"]
    dt = circ_info["dt"]
    order = circ_info["order"]
    res_ = []
    for step in range(steps):
        qc_ = trotter_2D_square(x_dim, y_dim, step, dt, J, h, hz, theta, order =order, meas = False, psi = False)
        basis_gates = [ 'rx', 'ry', 'rz', "rzz"]
        qc_ = transpile(qc_, basis_gates=  basis_gates, optimization_level=0)

        
        qasm = qiskit.qasm2.dumps(qc_)
        circ_ = qtn.Circuit.from_openqasm2_str(qasm)
        circ_.psi.apply_to_arrays(to_backend_)
        res_.append(complex(circ_.local_expectation(obs , where, optimize=opt)).real)
    return res_


def exact_obs_mps(psi, obs, where):
    to_backend, opt_, opt = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)
    psi = psi.copy()
    norm = (psi.H & psi).contract(all, optimize=opt)
    psi_ = psi.copy()
    
    qtn.tensor_network_gate_inds(psi_, to_backend_(obs) , [f"k{i}" for i in where], inplace=True) 
    val = (psi.H & psi_).contract(all, optimize=opt)

    
    return complex(val).real/ complex(norm).real



def simulate_mps(G, where, circ_info, max_bond=20, progbar=True, cutoff=1.e-16, steps=2):
    x_dim = circ_info["x_dim"]
    y_dim = circ_info["y_dim"]
    theta = circ_info["theta"]
    J = circ_info["J"]
    h = circ_info["h"]
    hz = circ_info["hz"]
    dt = circ_info["dt"]
    order = circ_info["order"]
    L = x_dim * y_dim
    res_ = []
    entropy_ = []

    to_backend, opt_, opt = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)
    basis_gates = [ 'rx', 'ry', 'rz', "rzz"]

    qc_ = trotter_2D_square(x_dim, y_dim, 1, dt, J, h, hz, 0, order = order)
    qc_ = transpile(qc_, basis_gates=  basis_gates, optimization_level=0)

    qc_0 = trotter_2D_square(x_dim, y_dim, 0, dt, J, h, hz, theta, order = order)
    qc_0 = transpile(qc_0, basis_gates=  basis_gates, optimization_level=0)

    
    psi = qtn.MPS_computational_state([0]*L)
    psi.apply_to_arrays(to_backend_)


    # step > 0
    for step_ in range(0, steps):
        if step_ == 0:
            qasm = qiskit.qasm2.dumps(qc_0)
            circ_ = qtn.Circuit.from_openqasm2_str(qasm)
        
            circ = qtn.CircuitMPS.from_gates(
                gates=circ_.gates,
                max_bond=max_bond,
                cutoff=cutoff,
                progbar=progbar,
                psi0=psi,
                to_backend = to_backend_,
                )   
            psi = circ.psi
            entropy = psi.entropy(L//2, method='svd')
            entropy_.append(entropy)
            val = exact_obs_mps(psi, G, where)
            res_.append(val)

        qasm = qiskit.qasm2.dumps(qc_)
        circ_ = qtn.Circuit.from_openqasm2_str(qasm)
        circ = qtn.CircuitMPS.from_gates(
            gates=circ_.gates,
            max_bond=max_bond,
            cutoff=cutoff,
            progbar=progbar,
            psi0=psi,
            to_backend = to_backend_,
            )   

        psi = circ.psi
        entropy = psi.entropy(L//2, method='svd')
        entropy_.append(entropy)
        val = exact_obs_mps(psi, G, where)
        res_.append(val)
    
    
    return res_, entropy_


def sim_mps_perm(G, where, circ_info, steps = 2, max_bond=20, progbar=True, cutoff=1.e-16, psi0=None,):
    x_dim = circ_info["x_dim"]
    y_dim = circ_info["y_dim"]
    theta = circ_info["theta"]
    J = circ_info["J"]
    h = circ_info["h"]
    hz = circ_info["hz"]
    dt = circ_info["dt"]
    order = circ_info["order"]
    L = x_dim * y_dim
    res_ = []
    entropy_ = []

    to_backend, opt_, opt = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)
    basis_gates = [ 'rx', 'ry', 'rz', "rzz"]

    qc_ = trotter_2D_square(x_dim, y_dim, steps, dt, J, h, hz, theta, order = order)
    qc_ = transpile(qc_, basis_gates=  basis_gates, optimization_level=0)

    
    qasm = qiskit.qasm2.dumps(qc_)
    circ_ = qtn.Circuit.from_openqasm2_str(qasm)
    circ = qtn.CircuitPermMPS.from_gates(
        gates = circ_.gates,
        max_bond=max_bond,
        cutoff=cutoff,
        progbar=progbar,
        #psi0=psi0,
        to_backend = to_backend_,
    )

    psi = circ.psi
    val = exact_obs_mps(psi, G, where)
    
    return val


def sim_mps_perm(G, where, circ_info, steps = 2, max_bond=20, progbar=True, cutoff=1.e-16, psi0=None,):
    x_dim = circ_info["x_dim"]
    y_dim = circ_info["y_dim"]
    theta = circ_info["theta"]
    J = circ_info["J"]
    h = circ_info["h"]
    hz = circ_info["hz"]
    dt = circ_info["dt"]
    order = circ_info["order"]
    L = x_dim * y_dim
    res_ = []
    entropy_ = []

    to_backend, opt_, opt = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)
    basis_gates = [ 'rx', 'ry', 'rz', "rzz"]

    qc_ = trotter_2D_square(x_dim, y_dim, steps, dt, J, h, hz, theta, order = order)
    qc_ = transpile(qc_, basis_gates=  basis_gates, optimization_level=0)

    
    qasm = qiskit.qasm2.dumps(qc_)
    circ_ = qtn.Circuit.from_openqasm2_str(qasm)
    circ = qtn.CircuitPermMPS.from_gates(
        gates = circ_.gates,
        max_bond=max_bond,
        cutoff=cutoff,
        progbar=progbar,
        #psi0=psi0,
        to_backend = to_backend_,
    )

    psi = circ.psi
    val = exact_obs_mps(psi, G, where)
    
    return val


def simulate_mps_(G_, where_G_, circ_info, max_bond=20, steps=4, 
                  cutoff=1.e-12,
                  t_im=0.0,
                  info_c = None,   #{"cur_orthog":"calc"},
                  cal_fidel = True,
                  normalize_ = True,
                  cal_entropy = False, guess_=False, 
                  cal_z2 = True, progdmrg = False,
                  method="mpo", # swap, mpo, dmrg
                  progbar=True, n_iter=10,  threshold = 1.e-8, opt=None,
                 ):

    x_dim = circ_info["x_dim"]
    y_dim = circ_info["y_dim"]
    theta = circ_info["theta"]
    J = circ_info["J"]
    h = circ_info["h"]
    hz = circ_info["hz"]
    dt = circ_info["dt"]
    order = circ_info["order"]
    L = x_dim * y_dim
    to_backend, _, _ = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)

    basis_gates = [ 'rx', 'ry', 'rz', "rzz"]

    qc_ = trotter_2D_square(x_dim, y_dim, 1, dt, J, h, hz, 0, order = order)
    #qc_ = transpile(qc_, basis_gates=  basis_gates, optimization_level=0)

    qc_0 = trotter_2D_square(x_dim, y_dim, 0, dt, J, h, hz, theta, order = order)
    qc_0 = transpile(qc_0, basis_gates=  basis_gates, optimization_level=0)

    # build-up the circuit init: |cos> + |sin>
    gate_l_0, where_l_0  = circ_gates(qc_0, 1, t_im=0)

    # build-up the repeated circuit
    gate_l, where_l  = circ_gates(qc_, steps, t_im=t_im, dt=dt)


    gate_l_0 = quf.gate_to_backend(gate_l_0, to_backend_)
    gate_l = quf.gate_to_backend(gate_l, to_backend_)

    gate_l = gate_l_0 + gate_l
    where_l = where_l_0 + where_l


    L = qc_.num_qubits
    p = qtn.MPS_computational_state([0]*L)
    if method=="dmrg":
        p.expand_bond_dimension(max_bond, inplace=True)
    
    p.apply_to_arrays(to_backend_)
    res_obs = []
    res_z2 = []
    entropy_ = []
    fidel_appro = [1]
    F_l = [1] #local fidelity
    F_l_ = []
    norm_l = []
    if info_c:
        p.canonicalize_([0], cur_orthog='calc', info=info_c)
    else:
        info_c = None
    p_l = []
    z2 = 0
    from tqdm.notebook import tqdm

    with tqdm(total=len(gate_l),  desc="mps:", leave=True, position=0, 
            colour='CYAN', disable = not progbar) as pbar:

        for count in range(len(gate_l)):

            # store mps in a list
            p_l.append(p.copy())

            for count_ in range(len(gate_l[count])):
                where = where_l[count][count_]
                G = gate_l[count][count_]

                if method=="swap":
                    if len(where)==1:
                        p_ = p.gate(G, where, contract=False)
                        p.gate_(G, where, contract=True, inplace=True)
                        
                    else:
                        if cal_fidel:
                            # uncompresed MPS
                            p_ = p.gate(G, where, contract=False, inplace=False)

                        norm_0 = p.norm()
                        p.gate_with_auto_swap_(G, where, info=info_c, max_bond = max_bond, swap_back=True, **{"cutoff":cutoff})
                        norm_1 = p.norm()
                        fidel_appro.append( complex(min(abs(norm_1/norm_0),abs(norm_0/norm_1))**2).real )

                    if cal_fidel:
                        # <ideal MPS | compressed MPS>
                        F_l.append( complex(quf.fidel_mps(p, p_, opt)).real )

                
                if method=="mpo":

                    if len(where)==1:
                        p_ = p.gate(G, where, contract=False)
                        p.gate_(G, where, contract=True, inplace=True)
                        
                    else:
                        if cal_fidel:
                            # uncompresed MPS
                            p_ = p.gate(G, where, contract=False)

                        norm_0 = p.norm()
                        p.gate_nonlocal_(G, where, max_bond = max_bond , info=info_c , method='direct', **{"cutoff":cutoff})
                        norm_1 = p.norm()
                        fidel_appro.append( complex(min(abs(norm_1/norm_0),abs(norm_0/norm_1))**2).real )

                    if cal_fidel:
                        # <ideal MPS | compressed MPS>
                        F_l.append( complex(quf.fidel_mps(p, p_, opt)).real )
    
                if method=="dmrg":
                    if len(where)==1:
                        p.gate_(G, where, contract=True, inplace=True)
                        F_l.append( 1.0 )
                    else:
                        p_ = p.gate(G, where, contract=False)
                        norm_0 = p.norm()

                        if guess_:
                            p.gate_nonlocal_(G, where, max_bond = max_bond , info=info_c , method='direct', **{"cutoff":cutoff})
                        
                        p, cost, cotengra_cost = quf.gate_dmrg(p, p_, opt,  n_iter=n_iter,  threshold = threshold, prgbar = progdmrg)
                        
                        norm_1 = p.norm()
                        fidel_appro.append( complex(min(abs(norm_1/norm_0),abs(norm_0/norm_1))**2).real )
                        F_l.append( complex(cost).real )

            
            z_l = []
            if info_c:
                if normalize_:
                    # normalize MPS with respect to canonalization site, i.e. cur_orthog:
                    p.normalize(insert=info_c["cur_orthog"][0])
                if cal_entropy:
                    entropy = p.entropy(L//2, cur_orthog=info_c["cur_orthog"], method='svd')
                    entropy_.append(entropy)

            else:
                if normalize_:
                    # normalize MPS:
                    p.normalize()
                if cal_entropy:
                    entropy = p.entropy(L//2, method='svd')
                    entropy_.append(entropy)

            norm_l.append(complex(p.norm()).real)

            # measure obs w.r.t MPS
            if where_G_:
                val = exact_obs_mps(p, G_, where_G_)
                res_obs.append(val)


            # Fidel = f * f * .... all steps
            if cal_fidel or method=="dmrg":
                f_ = np.prod(F_l)
                log_product = np.sum(np.log(F_l))
                f_ = np.exp(log_product)
            else:
                log_product = np.sum(np.log(fidel_appro))
                f_ = np.exp(log_product)
            
            F_l_.append(f_)
    

            if cal_z2:
                # measure <Z^2>: first put it into MPO:
                #mpoz2 = quf.mpo_zz_center(L, "Z", where_=L//2)
                mpoz2 = mpo_z2_(circ_info, "Z", where_=L//2, form="left")

                mpoz2.apply_to_arrays(to_backend_)
                z2 = complex(quf.energy_global(mpoz2, p)).real
                res_z2.append(z2)
                
            pbar.set_postfix({ "obs": val,
                              "error": 1-F_l_[-1],
                              "bnd":p.max_bond(),
                              "z2": z2,
                              })
            pbar.refresh()
            pbar.update(1)


            
    #collect results:
    res_ = {"fidel":F_l_, "p_l":p_l, "norm": norm_l, "obs":res_obs, "z2":res_z2, "entropy":entropy_   }
    #res_ |= {"gate_l":gate_l, "where_l":where_l}
    return res_





def simulate_peps_(obs, where_G_, circ_info, cluster=[], max_bond=4, prog_compress=False, steps=4, 
                   cutoff=1.e-12, t_im=0.0, cycle_peps = True,
                  compress_ = "bp",
                  ):

    x_dim = circ_info["x_dim"]
    y_dim = circ_info["y_dim"]
    theta = circ_info["theta"]
    J = circ_info["J"]
    h = circ_info["h"]
    hz = circ_info["hz"]
    dt = circ_info["dt"]
    order = circ_info["order"]
    L = x_dim * y_dim
    to_backend, opt_, opt = quf.requirement_to_backend()
    to_backend_ = quf.get_to_backend(to_backend)
    
    peps = quf.peps_I(x_dim, y_dim, theta = 0)
    if cycle_peps:
        peps = quf.peps_cycle(peps, int(1))
    
    
    

    basis_gates = [ 'rx', 'ry', 'rz', 'rzz']

    qc_ = trotter_2D_square(x_dim, y_dim, 1, dt, J, h, hz, 0, order = order)
    #qc_ = transpile(qc_, basis_gates=  basis_gates, optimization_level=0)

    qc_0 = trotter_2D_square(x_dim, y_dim, 0, dt, J, h, hz, theta, order = order)
    qc_0 = transpile(qc_0, basis_gates=  basis_gates, optimization_level=0)
    

    # build-up the circuit init: |cos> + |sin>
    gate_l_0, where_l_0  = circ_gates(qc_0, 1, t_im=0)
    # build-up the repeated circuit
    gate_l, where_l  = circ_gates(qc_, steps, t_im=t_im, dt=dt)
    
   
    
    # build-up pepo, a bit complicate, can be simplified 
    where_l_2d = quf.rotate_to_2d_(where_l_0[0], x_dim, y_dim)
    pepo_l_0, bonds = quf.gate_pepo(gate_l_0[0], where_l_2d, x_dim, y_dim, cycle_peps)
    
    where_l_2d = quf.rotate_to_2d_(where_l[0], x_dim, y_dim)

    
    pepo_l, bonds = quf.gate_pepo(gate_l[0], where_l_2d, x_dim, y_dim, cycle_peps)
    
    pepo_l_0 = quf.l_to_backend(pepo_l_0, to_backend_) 
    pepo_l = quf.l_to_backend(pepo_l, to_backend_) 
    peps.apply_to_arrays(to_backend_)
    
    pepo_l = pepo_l_0 + pepo_l*steps
    
    gate_l = gate_l_0 + gate_l
    where_l = where_l_0 + where_l

    res_peps = quf.simulate_peps(peps, pepo_l, max_bond, 
                                                     **{"site_2d": where_G_, "compress_" : compress_,
                                                        "prog_compress": prog_compress, 
                                                        "obs":obs, "cluster":cluster, "gate_l":gate_l, "where_l":where_l},
                                                     )



    return res_peps




def circ_gates(qc, steps, t_im=0.0, dt=1):

    # build up raw gates and its places (qubit number)   
    # t_im: imaginary part of time: e^(-i*theta*t) ---> e^(-i*theta*t) * e^(theta*time) 

    qasm = qiskit.qasm2.dumps(qc)
    circuit_opts = {"gate_contract":'False'}
    circ_ = qtn.Circuit.from_openqasm2_str(qasm)


    gate_l = []
    where_l = []
    for gate in circ_.gates:
        if len(gate.qubits)==1:
            # param_ = \theta * dt
            param_ = gate.params[0]


            t_im_ = 0
            if dt>1.e-6:
                t_im_ = (param_/dt)*t_im

            # Overall imagirary part is going to be a damping operator: e^-tau*H 
            #t_im_ = -abs(t_im_)

            
            param_ = [complex(param_, t_im_)]
            gate_ = qtn.circuit.Gate(gate.label, param_, gate.qubits)
            gate_l.append(gate_.array)
            where_l.append(gate.qubits)
        else:
            # param_ = \theta * dt
            param_ = gate.params[0]

            #t_im_ = t_im
            t_im_ = 0
            if dt>1.e-6:
                t_im_ = (param_/dt)*t_im
                

            # Overall imagirary part is going to be a damping operator: e^-tau*H 
            #t_im_ = -abs(t_im_)
            
            param_ = [complex(param_, t_im_)]
            gate_ = qtn.circuit.Gate(gate.label, param_, gate.qubits)
            gate_l.append(gate_.array)
            where_l.append(gate.qubits)
    
    gate_l = [gate_l] * steps
    where_l = [where_l] * steps
        
    return  gate_l, where_l 
    
def mpo_z2_(circ_info, pauli, where_=0, dtype="complex128", chi=200, cutoff=1.0e-16, form="left"):
    
    Lx = circ_info["x_dim"]
    Ly = circ_info["y_dim"]
    L = Lx*Ly

    obs = weights(Lx,Ly)
    mpoz2 = qtn.MPO_identity(L, phys_dim=2)* 1/L
    for where_ in obs:
        x, y = where_
        mpo_z = quf.mpo_z_prod(L, pauli, where_= [0, y*Lx + x])
        mpoz2 += mpo_z * (obs[where_]/(L**2))
        mpoz2.compress(form, max_bond=chi, cutoff=cutoff )

    
    return  mpoz2

