import numpy as np
import itertools
import logging
import matplotlib.pyplot as plt

import time
import scipy.io as scio
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import scipy.io as scio
import seaborn as sns
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal
from scipy.special import gamma

from nf.flows import *
from nf.models import NormalizingFlowModel
sns.set()

try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def gen_data(n=512):
    return np.r_[np.random.randn(n // 2, 1) + np.array([2]),
                 np.random.randn(n // 2, 1) + np.array([-2])]


def StableVariable(m, alpha):
     V = np.pi/2 * (2*np.random.rand(m)-1)
     W = np.random.exponential(scale=1, size=m)
     y = np.sin(alpha * V) / (np.cos(V)**(1/alpha) ) * (np.cos( V*(1-alpha)) / W )**((1-alpha)/alpha)
     return y

def GeneratingData(T, dt, n_samples, X0):
    t = np.arange(0, T, dt)
    Nt = len(t)
    alpha = 1.5
    x0 = X0 * np.ones([n_samples])
    N = x0.size
    x = np.zeros((Nt, N))
    x[0, :] = x0
    for i in range(Nt-1):
        Ut = dt**(1/alpha) * StableVariable(N, alpha)
        
        #double-well systems with L\'evy motion
        x[i+1, :] = x[i, :] + (4*x[i, :] - 1*x[i, :]**3)*dt + 1*Ut 
    return t, x[-1,:]



if __name__ == "__main__":

    setup_seed(123)
    tis1 = time.perf_counter()
    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=5, type=int)
    argparser.add_argument("--flow", default="NSF_AR", type=str)
    argparser.add_argument("--iterations", default=200, type=int) ## dt=0.01, iterations=100.
    args = argparser.parse_args()


    T = 0.002
    dt = 0.001
    x_init = np.linspace(-2.5, 2.5, 21)
    drift = np.zeros(len(x_init))
    sigma = np.zeros(len(x_init))
    resampling_data = np.zeros([0,1])
    count = 0
    for x0 in x_init:
        
        time_set, position_x = GeneratingData(T, dt, 5000, x0)
        
        ##data preprocessing
        mu_x = np.mean(position_x)
        sigma_x = 1/3*np.std(position_x)
        position_x = (position_x - mu_x) / sigma_x
        
        P_x = np.reshape(position_x, position_x.size, order='C').reshape(-1, 1)
        x = torch.Tensor(P_x)
        flow = eval(args.flow)
        flows = [flow(dim=1) for _ in range(args.flows)]
        prior = MultivariateNormal(torch.zeros(1), torch.eye(1))
        model = NormalizingFlowModel(prior, flows)
    
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        
        
    
        loader = data.DataLoader(
        dataset = x,
        batch_size=50000,             
        shuffle=True,             
        )
        
        Loss = np.array([])
        for epoch in range(args.iterations):    
            for step, batch_x in enumerate(loader): 
                optimizer.zero_grad()
                z, prior_logprob, log_det = model(x)
                logprob = prior_logprob + log_det
                loss = -torch.mean(prior_logprob + log_det)
                loss.backward()
                optimizer.step()
                tmp = loss.cpu().detach().numpy()
                Loss = np.append(Loss, tmp)
        print("count:", count)

        ##System Identification
        ##data preprocessing
        a, b = ((-1.+x0) - mu_x) / sigma_x, ((1.+x0) - mu_x) / sigma_x
        u0 = np.linspace(a, b, 201)
        
        
        du = 2/200
        u1 = np.reshape(u0, u0.size, order='C').reshape(-1, 1)
        uu = torch.Tensor(u1)
        z, prior_logprob, log_det = model(uu)
        samples_px = torch.exp(prior_logprob + log_det)
        px_estimated = samples_px.cpu().detach().numpy()
        

        q=np.arange(0,len(Loss))
        plt.plot(q,Loss,'r')
        plt.show()
        
        
        p_learned = px_estimated 

        
        ## For drift
        integral = np.sum(p_learned*(u0*sigma_x+mu_x-x0))*du / (dt * sigma_x)
        drift[count] = integral
        
        ## for kernel
        M = 10000
        samples = model.sample(M).data.cpu().detach().numpy()
        samples = samples * sigma_x + mu_x - x0
        resampling_data = np.append(resampling_data, samples, axis=0)
        
        count += 1
        
    ## Plotting drift
    drift_coeff = np.polyfit(x_init, drift, 3)
    p1 = np.poly1d(drift_coeff)
    drift_fitting = p1(x_init)  
    plt.style.use('classic')
    plt.figure(figsize=(12,8), facecolor='white', edgecolor='black')
    
    l1, = plt.plot(x_init, drift_fitting, 'r', linewidth=3)
    drift_T = 4*x_init - x_init**3
    l2, = plt.plot(x_init, drift_T, 'b', linewidth=3)
    plt.xlabel("x1",fontsize=20)
    plt.ylabel("x2",fontsize=20)
    plt.legend(handles=[l1,l2],labels=['learned drift','true drift'],loc='upper right',fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(-3,3)
    plt.ylim(-6.,6.)
    plt.grid(False)
    plt.show()
    
    tis2 = time.perf_counter()
    print("Time used:", tis2-tis1)
    

