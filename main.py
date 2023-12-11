import numpy as np
import torch
import argparse
from utils import str2bool
from solver import Solver
import os
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

global_parameters = {
    "repetitions":   5,                            # total number of repetitions
    "beta":          list(np.logspace(-7, 1, 17)), # regularization parameter
    "loss_id":       [1,2,3],                      # 0 - no regularizarion, 1 - VIB with standard Gaussian prior,
                                                   # 2 - lossless CDVIB (ours), 3 - lossy CDVIB (ours)
    "centers_num":   [5],                          # number of centers in loss 2 and loss 3
    "mov_coeff_mul": [1e-4],                       # coefficient for smooth change of prior centers in loss 2 and loss 3
    "results_dir":   'results',                    # dir to results
    "figures_dir":   'figures',                    # dir to figures
    "save_model":    True                          # saves the model after training
}


solver_parameters = {
    "cuda":          True,
    "seed":          0,                   # used to re-initialize dataloaders 
    "epoch_1":       200,                 # first training phase
    "epoch_2":       100,                 # second training phase                                          
    "lr":            1e-4,                # learning rate
    "beta":          0,                   # regularization parameter
    "alpha":         [0.1],               # relaxation coefficient
    "K":             64,                  # dimension of encoding Z    
    "num_avg":       12,                  # number of samplings Z
    "batch_size":    128,                 # batch size
    "dataset":       'CIFAR10',           # dataset
    "dset_dir":      'datasets',          # dir with datasets
    "per_epoch_stat": False               # true if some statistics computed every epoch (slow) 
}


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)

    # create a new folder to store our results and figures
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = global_parameters["results_dir"] + '/results_' +  timestamp
    os.makedirs(results_path)
    figures_path = global_parameters["figures_dir"] + '/figures_' +  timestamp
    os.makedirs(figures_path)
    # save global and initial solver parameters to a file
    np.savez(results_path + '/global_parameters', global_parameters = global_parameters)          

    # training of models with various parameters
    for rep in range(global_parameters["repetitions"]):
        for loss_id in global_parameters["loss_id"]:
            for beta in global_parameters["beta"]:
                for centers_num in global_parameters["centers_num"]:
                    for mov_coeff_mul in global_parameters["mov_coeff_mul"]:                        
                        # update selected solver parameters
                        solver_parameters["seed"]          = rep
                        solver_parameters["beta"]          = beta
                        solver_parameters["loss_id"]       = loss_id
                        solver_parameters["mov_coeff_mul"] = mov_coeff_mul
                        solver_parameters["centers_num"]   = centers_num
                        
                        # create a model and train
                        net = Solver(solver_parameters)
                        net.train_full()
                        # extract interesting statistics and save to a .npz file
                        filename = '/_results_LossID_{}_Beta_{:,.0e}_NumCent_{}_MovCoeff_{}_rep_{}_.npz'.format(loss_id,beta,centers_num,mov_coeff_mul,rep)
                        np.savez(results_path+filename, 
                                    solver_parameters    = solver_parameters,
                                    train1_train_dataset = net.train1_train_dataset, 
                                    train1_test_dataset  = net.train1_test_dataset,
                                    train2_train_dataset = net.train2_train_dataset,
                                    train2_test_dataset  = net.train2_test_dataset,
                                    train1_epochs        = net.train1_epochs,
                                    moving_average_mul   = net.moving_mean_multiple_tensor.cpu().numpy(),
                                    moving_variance_mul  = net.moving_variance_multiple_tensor.cpu().numpy(),
                                    counter              = rep
                            )
                        
                        if global_parameters["save_model"]: torch.save(net.IBnet, results_path+'/_trained_model_LossID_{}_Beta_{:,.0e}_NumCent_{}_MovCoeff_{}_rep_{}_.pth'.format(loss_id,beta,centers_num,mov_coeff_mul,rep))
                        del net 

if __name__ == "__main__":
    main()
