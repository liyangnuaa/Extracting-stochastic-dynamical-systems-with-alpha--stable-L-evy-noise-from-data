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
from argparse import ArgumentParser
from torch.distributions import MultivariateNormal
from sympy import *
from scipy.special import gamma

from nf.flows import *
from nf.models import NormalizingFlowModel

from scipy import integrate






try:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # enable for GPU
except:
    pass
def setup_seed(seed):
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      np.random.seed(seed)
      torch.backends.cudnn.deterministic = True



    

if __name__ == "__main__":
    setup_seed(123) 
    tis1 = time.perf_counter()
    argparser = ArgumentParser()
    argparser.add_argument("--n", default=512, type=int)
    argparser.add_argument("--flows", default=1, type=int)
    argparser.add_argument("--flow", default="RealNVP", type=str)
    argparser.add_argument("--iterations", default=200, type=int)
    argparser.add_argument("--use-mixture", action="store_true")
    argparser.add_argument("--convolve", action="store_true")
    argparser.add_argument("--actnorm", action="store_true")
    args = argparser.parse_args()


    x_init = np.linspace(-2, 2, 6)
    y_init = np.linspace(-2, 2, 6)
    x_init_grid, y_init_grid = np.meshgrid(x_init, y_init)
    data_init = np.concatenate((x_init_grid.reshape(-1,1), y_init_grid.reshape(-1,1)), axis=1)
    drift1 = np.zeros(len(data_init))
    drift2 = np.zeros(len(data_init))
    sigma1 = np.zeros(len(data_init))
    sigma2 = np.zeros(len(data_init))
    resampling_data = np.zeros([0,2])
    count = 0
    for (x0, y0) in data_init:
        
        flow = eval(args.flow)
        flows = [flow(dim=2) for _ in range(args.flows)]
        prior = MultivariateNormal(torch.zeros(2), 1*torch.eye(2))
        model = NormalizingFlowModel(prior, flows)
        optimizer = optim.Adam(model.parameters(), lr=0.005)
        

        ## loading a SDE data
        count_tmp = count + 1
        dataFile = "Data_"+ "%d"  %count_tmp
        SDE_data = scio.loadmat(dataFile)
        position_x = SDE_data['x_end']
        position_y = SDE_data['y_end']
        
        ##data preprocessing
        mu_x, mu_y = np.mean(position_x), np.mean(position_y)
        sigma_x, sigma_y = 1/3 * np.std(position_x), 1/3 * np.std(position_y)
        position_x = (position_x - mu_x) / sigma_x
        position_y = (position_y - mu_y) / sigma_y
        
        P_x = np.reshape(position_x, position_x.size, order='C').reshape(-1, 1)
        P_y = np.reshape(position_y, position_y.size, order='C').reshape(-1, 1)
        
        x = torch.Tensor(np.concatenate((P_x,P_y),axis=1))
        
        
        
        
        loader = data.DataLoader(
        dataset = x,
        batch_size=5000,
        shuffle=True,            
        )
        Loss = np.array([])
        rho = 1
        for epoch in range(args.iterations):    
            for step, batch_x in enumerate(loader): 
                optimizer.zero_grad()
                z, prior_logprob, log_det = model(x)
                logprob = prior_logprob + log_det
                loss = -torch.mean(prior_logprob + log_det) + 0*np.log(rho) - 0*torch.log(torch.std(logprob)) + 0*torch.std(logprob)
                loss.backward()
                optimizer.step()
                tmp = loss.cpu().detach().numpy()
                Loss = np.append(Loss, tmp)
        q=np.arange(0,len(Loss))
        plt.plot(q,Loss,'r')
        plt.show()
        print("count:", count)


        ##System Identification
        a, b = ((-0.3+x0) - mu_x) / sigma_x, ((0.3+x0) - mu_x) / sigma_x
        c, d = ((-0.3+y0) - mu_y) / sigma_y, ((0.3+y0) - mu_y) / sigma_y
        u0, v0 = np.linspace(a, b, 301), np.linspace(c, d, 301)


        du = 0.6/300
        dv = 0.6/300
        u, v = np.meshgrid(u0, v0)
        u1 = np.reshape(u, u.size, order='C').reshape(-1, 1)
        v1 = np.reshape(v, v.size, order='C').reshape(-1, 1)
        uu = torch.Tensor(np.concatenate((u1,v1),axis=1))
        t_star = 0.001
        z, prior_logprob, log_det = model(uu)
        samples_px = torch.exp(prior_logprob + log_det)
        px_estimated = samples_px.cpu().detach().reshape(u.shape).numpy()
        
        

        
        
        
        # ## for kernel
        # M = 10000
        # samples = model.sample(M).data.cpu().detach().numpy()
        # samples[:,0] = samples[:,0] * sigma_x + mu_x - x0
        # samples[:,1] = samples[:,1] * sigma_y + mu_y - y0
        # resampling_data = np.append(resampling_data, samples, axis=0)


        
        
        
        
        
        
        ## for drift
        tmp1 = (u0*sigma_x+mu_x-x0)*px_estimated
        tmp2 = (v0*sigma_y+mu_y-y0)*(px_estimated.T)
        

        
        tmp1_j = np.r_[tmp1[1:, :], np.zeros([1, tmp1[0,:].size])]
        tmp1_i = np.c_[tmp1[:, 1:], np.zeros([tmp1[:, 0].size, 1])] 
        tmp1_ij_tmp = np.r_[tmp1[1:, 1:], np.zeros([1, tmp1[0,:].size-1])]
        tmp1_ij = np.c_[tmp1_ij_tmp, np.zeros([tmp1[:, 0].size, 1])]
        
        tmp2_j = np.r_[tmp2[1:, :], np.zeros([1, tmp2[0,:].size])]
        tmp2_i = np.c_[tmp2[:, 1:], np.zeros([tmp2[:, 0].size, 1])] 
        tmp2_ij_tmp = np.r_[tmp2[1:, 1:], np.zeros([1, tmp2[0,:].size-1])]
        tmp2_ij = np.c_[tmp2_ij_tmp, np.zeros([tmp2[:, 0].size, 1])]
        drift1[count] = du*dv*np.sum((tmp1 + tmp1_j + tmp1_i + tmp1_ij)) / (4*t_star * sigma_x * sigma_y)
        drift2[count] = du*dv*np.sum((tmp2 + tmp2_j + tmp2_i + tmp2_ij)) / (4*t_star * sigma_x * sigma_y)
        
        count += 1


    tis2 = time.perf_counter()
    print("Time used:", tis2-tis1)
    
    ##Plotting coefficients
    drift1_learned = np.reshape(drift1, x_init_grid.shape)
    drift2_learned = np.reshape(drift2, x_init_grid.shape)
    drift1_true = 5*x_init_grid - y_init_grid**2
    drift2_true = 5*x_init_grid + y_init_grid



    plt.figure(figsize=(27,17), facecolor='white', edgecolor='black')
    plt.subplot(2, 2, 1)
    c1 = plt.pcolormesh(x_init_grid, y_init_grid, drift1_true, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("drift1",fontsize=25)
    plt.colorbar(c1)
    
    plt.subplot(2, 2, 2)
    c2 = plt.pcolormesh(x_init_grid, y_init_grid, drift2_true, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("drift2",fontsize=25)
    plt.colorbar(c2)
    
    plt.subplot(2, 2, 3)
    c5 = plt.pcolormesh(x_init_grid, y_init_grid, drift1_learned, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(c1)
    
    plt.subplot(2, 2, 4)
    c6 = plt.pcolormesh(x_init_grid, y_init_grid, drift2_learned, cmap='jet', shading='gouraud')
    plt.xlabel("x1",fontsize=25)
    plt.ylabel("x2",fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.colorbar(c2)
    


    # ## for kernel
    # m, eps = 2, 0.5 ## for coupling multi BM+LM 
    # N = 1
    # M = 36 * 10000
    # alpha_estimated = np.zeros(N)
    # sigma_estimated = np.zeros(N)
    # for k in range(N):
    #     k = k + 1
        
    #     result1 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (eps)**2
    #     result2 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (m * eps)**2
    #     n_0 = np.sum(result2) - np.sum(result1)
    #     result3 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (m**k * eps)**2
    #     result4 = pow(resampling_data[:,0], 2) + pow(resampling_data[:,1], 2) < (m**(k+1) * eps)**2
    #     n_k = np.sum(result4) - np.sum(result3)
        
        
    #     alpha_estimated[k-1] = 1/(k*np.log(m)) * np.log(n_0/n_k)
    
    
    #     ## Estimating  sigma
    #     tmp1 = alpha_estimated[k-1] * eps**alpha_estimated[k-1] * m**(k*alpha_estimated[k-1]) *n_k
    #     C_alpha = alpha_estimated[k-1]*gamma(1+alpha_estimated[k-1]/2) / (2**(1-alpha_estimated[k-1])*np.pi*gamma(1-alpha_estimated[k-1]/2))
    #     tmp2 = 2*np.pi * C_alpha * t_star * M * (1 - 1/m**alpha_estimated[k-1])
    #     sigma_estimated[k-1] = (tmp1/tmp2)**(1/alpha_estimated[k-1])


    print('alpha:',np.average(alpha_estimated))
    print('sigma:',np.average(sigma_estimated))



        
