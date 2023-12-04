import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from datasets import return_data
from utils import cuda
from model import IBNet
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st
import random
 
class Solver(object):

    def __init__(self, args):
        self.args = args

        # Training parameters
        self.cuda         = (args["cuda"] and torch.cuda.is_available())
        self.seed         = args["seed"]
        self.epoch_1      = args["epoch_1"]
        self.epoch_2      = args["epoch_2"]
        self.batch_size   = args["batch_size"]
        self.lr           = args["lr"]
        self.eps          = 1e-15
        self.global_iter  = 0
        self.global_epoch = 0

        # Model parameters
        self.K       = args["K"]
        self.beta    = args["beta"]
        self.alpha   = args["alpha"]
        self.num_avg = args["num_avg"]
        self.loss_id = args["loss_id"] 
    
        # CDVIB parameters
        self.M       = args["centers_num"] 
        self.moving_coefficient_multiple = args["mov_coeff_mul"]
        self.moving_mean_multiple_tensor     = cuda(torch.zeros(10*self.M,self.K),self.cuda)
        self.moving_variance_multiple_tensor = cuda(torch.ones(10*self.M,self.K),self.cuda)

        # Networks
        self.IBnet = cuda(IBNet(self.K), self.cuda)
        self.IBnet.weight_init()
        
        # Optimizer
        self.optim     = optim.Adam(self.IBnet.parameters(),lr=self.lr,betas=(0.5,0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)
        
        # Dataset
        self.data_loader = return_data(args["dataset"], args["dset_dir"], args["batch_size"])
        if args["dataset"] == 'MNIST':   self.HY = math.log(10,2)
        if args["dataset"] == 'CIFAR10': self.HY = math.log(10,2)

        # Other
        self.per_epoch_stat = args["per_epoch_stat"]
    

    def set_mode(self,mode='train'):
        if mode == 'train' :
            self.IBnet.train()
            self.mode = 'train'
        elif mode == 'eval' :
            self.IBnet.eval()
            self.mode='eval'
        else : raise('mode error. It should be either train or eval')


    def regularization(self, mu, std, y, idx, reduction = 'mean'):
        if idx == 0:   # no regularization
            info_loss = cuda(torch.tensor(0.0),self.cuda)
        elif idx == 1: # standard VIB regularization with Gaussian prior
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum().div(math.log(2))          
        elif idx == 2: # lossless CDVIB
            # select closest centers
            centers_mean_label = self.moving_mean_multiple_tensor.view(10,self.M,self.K)[y,:,:]     # size [B, M, K]
            centers_var_label  = self.moving_variance_multiple_tensor.view(10,self.M,self.K)[y,:,:] # size [B, M, K]
            centers_selected_ind  = (centers_var_label.log() \
                                    + (centers_mean_label-mu.unsqueeze(1).repeat(1,self.M,1)).pow(2)/(centers_var_label+self.eps) \
                                    + std.pow(2).unsqueeze(1).repeat(1,self.M,1)/(centers_var_label+self.eps)).sum(2).argmin(dim=1) \
                                    + y*self.M                                                      # size [B]
            center_mean_selected = self.moving_mean_multiple_tensor[centers_selected_ind,:]         # size [B, K]
            center_var_selected  = self.moving_variance_multiple_tensor[centers_selected_ind,:]     # size [B, K]
            # info loss
            info_loss = (- 0.5*(1+2*std.log()) \
                        + 0.5 * center_var_selected.log() \
                        + 0.5 * (center_mean_selected-mu).pow(2)/(center_var_selected+self.eps) \
                        + 0.5 * std.pow(2)/(center_var_selected+self.eps)).sum().div(math.log(2))   # size [1]
            # update centers
            if self.mode == 'train':
                centers_selected_hot_encoding = torch.nn.functional.one_hot(centers_selected_ind,10*self.M).type(torch.float).detach() # size [B,C*M]
                center_count = centers_selected_hot_encoding.sum(0)
                center_mean_batch = torch.matmul(centers_selected_hot_encoding.transpose(0,1), mu).detach()                            # size [C*M,K] 
                center_var_batch  = torch.matmul(centers_selected_hot_encoding.transpose(0,1), std.pow(2)).detach()                    # size [C*M,K]

                self.moving_mean_multiple_tensor     *= (1 - self.moving_coefficient_multiple * self.M * center_count.unsqueeze(1).repeat(1,self.K))
                self.moving_variance_multiple_tensor *= (1 - self.moving_coefficient_multiple * self.M * center_count.unsqueeze(1).repeat(1,self.K))
                self.moving_mean_multiple_tensor     += self.moving_coefficient_multiple * self.M * center_mean_batch
                self.moving_variance_multiple_tensor += self.moving_coefficient_multiple * self.M * center_var_batch

        elif idx == 3: # lossy CDVIB
            # select closest centers
            centers_mean_label = self.moving_mean_multiple_tensor.view(10,self.M,self.K)[y,:,:]     # size [B, M, K]
            centers_var_label  = self.moving_variance_multiple_tensor.view(10,self.M,self.K)[y,:,:] # size [B, M, K]
            centers_selected_ind  = (centers_var_label.log() \
                                    + (centers_mean_label-mu.unsqueeze(1).repeat(1,self.M,1)).pow(2) \
                                    + std.pow(2).unsqueeze(1).repeat(1,self.M,1)/(centers_var_label+self.eps)).sum(2).argmin(dim=1) \
                                    + y*self.M                                                      # size [B]
            center_mean_selected = self.moving_mean_multiple_tensor[centers_selected_ind,:]         # size [B, K]
            center_var_selected  = self.moving_variance_multiple_tensor[centers_selected_ind,:]     # size [B, K]
            # info loss
            info_loss = (- 0.5*(1+2*std.log()) \
                        + 0.5 * center_var_selected.log() \
                        + 0.5 * (center_mean_selected-mu).pow(2) \
                        + 0.5 * std.pow(2)/(center_var_selected+self.eps)).sum().div(math.log(2))   # size [1]
            # update centers
            if self.mode == 'train':
                centers_selected_hot_encoding = torch.nn.functional.one_hot(centers_selected_ind,10*self.M).type(torch.float).detach() # size [B,C*M]
                center_count = centers_selected_hot_encoding.sum(0)
                center_mean_batch = torch.matmul(centers_selected_hot_encoding.transpose(0,1), mu).detach()                            # size [C*M,K] 
                center_var_batch  = torch.matmul(centers_selected_hot_encoding.transpose(0,1), std.pow(2)).detach()                    # size [C*M,K]

                self.moving_mean_multiple_tensor     *= (1 - self.moving_coefficient_multiple * self.M * center_count.unsqueeze(1).repeat(1,self.K))
                self.moving_variance_multiple_tensor *= (1 - self.moving_coefficient_multiple * self.M * center_count.unsqueeze(1).repeat(1,self.K))
                self.moving_mean_multiple_tensor     += self.moving_coefficient_multiple * self.M * center_mean_batch
                self.moving_variance_multiple_tensor += self.moving_coefficient_multiple * self.M * center_var_batch
            
        if reduction == 'sum':
            return info_loss
        elif reduction == 'mean':
            return info_loss.div(y.size(0))


    def train_full(self):
        print('Loss ID:{}, Beta:{:.0e}, NumCent:{}, MovCoeff:{}'.format(self.loss_id,self.beta,self.M,self.moving_coefficient_multiple))
        ##################
        # First training #
        ##################
        # reinitialize seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # train
        self.train_step1()
        # Computing some statistics after the fist training
        # a) accuracy over the training/testing datasets OK
        # b) average log-likelihood over the training/testing datasets OK
        # c) relevance I(Z;Y) and complexity I(Z;X) and regularization complexity OK
        # d) confidence intervals for accuracy/log-likelihood over the training/testing datasets OK
        # e) relevance I(Z;Y) and complexity I(Z;X) and regularization complexity per epoch OK
        print("Training 1 is finished")
        self.train1_epochs = {} 
        if self.per_epoch_stat == True:
                self.train1_epochs["izy_relevance_epochs"]  = self.izy_relevance_epochs.cpu().numpy()
                self.train1_epochs["izx_bound"] = self.izx_complexity_epochs.cpu().numpy()
                self.train1_epochs["reg_complexity"] = self.reg_complexity_epochs.cpu().numpy()
        self.test('train')
        self.train1_train_dataset = {}
        self.train1_train_dataset["accuracy"] = self.accuracy.cpu().numpy()
        self.train1_train_dataset["accuracy_confidence_low"] = self.accuracy_confidence_intervals[0]
        self.train1_train_dataset["accuracy_confidence_high"] = self.accuracy_confidence_intervals[1]

        self.train1_train_dataset["log_likelihood"] = self.log_likelihood.cpu().numpy()
        self.train1_train_dataset["log_likelihood_confidence_low"] = self.log_likelihood_confidence_intervals[0]
        self.train1_train_dataset["log_likelihood_confidence_high"] = self.log_likelihood_confidence_intervals[1]

        self.train1_train_dataset["izy_bound"] = self.izy_bound.cpu().numpy()
        self.train1_train_dataset["izx_bound"] = self.izx_bound.cpu().numpy()
        self.train1_train_dataset["reg_complexity"] = self.reg_complexity.cpu().numpy()

        self.test('test')
        self.train1_test_dataset = {}
        self.train1_test_dataset["accuracy"] = self.accuracy.cpu().numpy()
        self.train1_test_dataset["accuracy_confidence_low"] = self.accuracy_confidence_intervals[0]
        self.train1_test_dataset["accuracy_confidence_high"] = self.accuracy_confidence_intervals[1]

        self.train1_test_dataset["log_likelihood"] = self.log_likelihood.cpu().numpy()
        self.train1_test_dataset["log_likelihood_confidence_low"] = self.log_likelihood_confidence_intervals[0]
        self.train1_test_dataset["log_likelihood_confidence_high"] = self.log_likelihood_confidence_intervals[1]

        self.train1_test_dataset["izy_bound"] = self.izy_bound.cpu().numpy()
        self.train1_test_dataset["izx_bound"] = self.izx_bound.cpu().numpy()
        self.train1_test_dataset["reg_complexity"] = self.reg_complexity.cpu().numpy()
        print("Testing 1 is finished")

        ###################
        # Second training #
        ###################
        self.train2_train_dataset, self.train2_test_dataset = {}, {}
        self.train2_train_dataset["accuracy"], self.train2_test_dataset["accuracy"] = [], []
        self.train2_train_dataset["accuracy_confidence_low"], self.train2_test_dataset["accuracy_confidence_low"] = [], []
        self.train2_train_dataset["accuracy_confidence_high"], self.train2_test_dataset["accuracy_confidence_high"] = [], []
        self.train2_train_dataset["log_likelihood"], self.train2_test_dataset["log_likelihood"] = [], []
        self.train2_train_dataset["log_likelihood_confidence_low"], self.train2_test_dataset["log_likelihood_confidence_low"] = [], []
        self.train2_train_dataset["log_likelihood_confidence_high"], self.train2_test_dataset["log_likelihood_confidence_high"] = [], []
        self.train2_train_dataset["izy_bound"], self.train2_test_dataset["izy_bound"] = [], []
        self.train2_train_dataset["izx_bound"], self.train2_test_dataset["izx_bound"] = [], []
        self.train2_train_dataset["reg_complexity"], self.train2_test_dataset["reg_complexity"] = [], []
        # reinitialize seeds
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        # train
        for alpha in self.alpha:
            print('K:{}, Beta:{:.0e}, Correlation:{:,.3f}, Loss ID:{}, Alpha:{:,.3f}'.format(self.K,self.beta,self.corr,self.loss_id,alpha))
            # re-initialize schedulers
            self.optim     = optim.Adam(self.IBnet.parameters(),lr=self.lr,betas=(0.5,0.999))
            self.scheduler = lr_scheduler.ExponentialLR(self.optim,gamma=0.97)
            self.train_step2(alpha)
            print("Training 2 with alpha={:,.3f} is finished".format(alpha))
            # Computing some statistics after the fist training OK
            # a) accuracy over the training/testing datasets OK
            # b) average log-likelihood over the training/testing datasets OK
            # c) relevance I(Z;Y) and complexity I(Z;X) OK
            # d) confidence intervals for accuracy/log-likelihood over the training/testing datasets OK
            self.test('train')
            self.train2_train_dataset["accuracy"].append(self.accuracy.cpu().numpy())
            self.train2_train_dataset["accuracy_confidence_low"].append(self.accuracy_confidence_intervals[0])
            self.train2_train_dataset["accuracy_confidence_high"].append(self.accuracy_confidence_intervals[1])

            self.train2_train_dataset["log_likelihood"].append(self.log_likelihood.cpu().numpy())
            self.train2_train_dataset["log_likelihood_confidence_low"].append(self.log_likelihood_confidence_intervals[0])
            self.train2_train_dataset["log_likelihood_confidence_high"].append(self.log_likelihood_confidence_intervals[1])

            self.train2_train_dataset["izy_bound"].append(self.izy_bound.cpu().numpy())
            self.train2_train_dataset["izx_bound"].append(self.izx_bound.cpu().numpy())
            self.train2_train_dataset["reg_complexity"].append(self.reg_complexity.cpu().numpy())

            self.test('test')
            self.train2_test_dataset["accuracy"].append(self.accuracy.cpu().numpy())
            self.train2_test_dataset["accuracy_confidence_low"].append(self.accuracy_confidence_intervals[0])
            self.train2_test_dataset["accuracy_confidence_high"].append(self.accuracy_confidence_intervals[1])

            self.train2_test_dataset["log_likelihood"].append(self.log_likelihood.cpu().numpy())
            self.train2_test_dataset["log_likelihood_confidence_low"].append(self.log_likelihood_confidence_intervals[0])
            self.train2_test_dataset["log_likelihood_confidence_high"].append(self.log_likelihood_confidence_intervals[1])

            self.train2_test_dataset["izy_bound"].append(self.izy_bound.cpu().numpy())
            self.train2_test_dataset["izx_bound"].append(self.izx_bound.cpu().numpy())
            self.train2_test_dataset["reg_complexity"].append(self.reg_complexity.cpu().numpy())
            print("Testing 2 with alpha={:,.3f} is finished".format(alpha))


    def train_step1(self):

        self.set_mode('train')

        self.izy_relevance_epochs, self.izx_complexity_epochs, self.reg_complexity_epochs = [], [], []

        for e in range(self.epoch_1) :
            self.global_epoch += 1
            print('epoch:{}'.format(e))

            if self.per_epoch_stat == True:
                total_num, class_loss_av, zx_complexity_av, reg_complexity_av = 0, 0, 0, 0
            
            for idx, (images,labels) in enumerate(self.data_loader['train']):
                self.global_iter += 1

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                (mu, std), logit = self.IBnet(x)

                class_loss = F.cross_entropy(logit,y).div(math.log(2))
                info_loss = self.regularization(mu, std, y, self.loss_id) 
                total_loss = class_loss + self.beta*info_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                # computing some statistics per batch
                if self.per_epoch_stat == True:
                    total_num += y.size(0)
                    (mu, std), _ = self.IBnet(x,self.num_avg)
                    class_loss_av += class_loss.detach()
                    reg_complexity_av += info_loss.detach()
                    zx_complexity_av  += self.regularization(mu, std, y, 1).detach()

            # computing some statistics per epoch
            if self.per_epoch_stat == True:
                self.izy_relevance_epochs.append(self.HY - class_loss_av/idx)
                self.izx_complexity_epochs.append(reg_complexity_av/idx)
                self.reg_complexity_epochs.append(zx_complexity_av/idx)

            # print('class_loss:{:0.2f}, info_loss:{:0.2f}, total_loss:{:,.2f}'.format(class_loss,info_loss,total_loss))
            
            self.scheduler.step()


    def train_step2(self, alpha = 0.1):

        self.set_mode('train')

        # freeze the encoder
        for param in self.IBnet.encode.parameters():
            param.requires_grad_(False)
        # re-train the decoder 
        for e in range(self.epoch_2) :

            self.global_epoch += 1
            print('epoch:{}'.format(e))
            
            for ((images_train,labels_train), (images_test,labels_test))  in zip(self.data_loader['train'], self.data_loader['test_bootstrap']):
                self.global_iter += 1

                x_train = Variable(cuda(images_train, self.cuda))
                y_train = Variable(cuda(labels_train, self.cuda))
                _, logit = self.IBnet(x_train)
                class_loss_train = F.cross_entropy(logit,y_train).div(math.log(2))

                x_test = Variable(cuda(images_test, self.cuda))
                y_test = Variable(cuda(labels_test, self.cuda))
                _, logit = self.IBnet(x_test)
                class_loss_test = F.cross_entropy(logit,y_test).div(math.log(2))            
                
                total_loss = class_loss_train -alpha*class_loss_test

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                # computing some statistics per batch
                #
            # computing some statistics per epoch
            #       
            self.scheduler.step()    

        # unfreeze the encoder
        for param in self.IBnet.encode.parameters():
            param.requires_grad_(True)


    def test(self, dataloader_type, save_ckpt=True):
        self.set_mode('eval')
        total_num, correct, cross_entropy, zx_complexity, reg_complexity = cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda), cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda)
        for idx, (images,labels) in enumerate(self.data_loader[dataloader_type]):
            x = Variable(cuda(images, self.cuda))
            y = Variable(cuda(labels, self.cuda))
            total_num += y.size(0)
            (mu, std), soft_logit = self.IBnet(x,self.num_avg)
            if self.num_avg == 1 :
                # cross entropy
                cross_entropy += F.cross_entropy(soft_logit, y, reduction='sum').detach()
                # accuracy
                prediction = soft_logit.max(1)[1]
                correct += torch.eq(prediction,y).float().sum().detach()
            elif self.num_avg > 1:
                # cross entropy
                cross_entropy += sum(F.cross_entropy(soft_logit[j,:,:], y, reduction='sum').detach() for j in range(self.num_avg))
                # accuracy
                predictions = [soft_logit[j,:,:].max(1)[1] for j in range(self.num_avg)]
                correct += sum(torch.eq(predictions[j],y).float().sum().detach() for j in range(self.num_avg))
            # complexity
            zx_complexity  += self.regularization(mu, std, y, 1, reduction = 'sum').detach()
            reg_complexity += self.regularization(mu, std, y, self.loss_id, reduction = 'sum').detach()
        # some statistics at the end of testing
        self.accuracy       = correct/total_num/self.num_avg
        self.log_likelihood = -cross_entropy/total_num/self.num_avg
        self.izy_bound      = self.HY - cross_entropy/total_num/self.num_avg/math.log(2)
        self.izx_bound      = zx_complexity/total_num
        self.reg_complexity = reg_complexity/total_num
        self.bootstrap_confidence_intervals(dataloader_type = dataloader_type + '_bootstrap', confidence = 0.95, sample_size=1000, repetitions=100)


    def bootstrap_confidence_intervals(self, dataloader_type, confidence = 0.95, sample_size=1000, repetitions=100):
        accuracies, log_likelihoods_av = [], []
        # repeat repetitions time
        for rep in range(repetitions):
            total_num, correct, log_likelihood = cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda),cuda(torch.tensor(0,dtype=torch.float64),self.cuda)     
            # take randomly samples from the dataset
            for idx, (images,labels) in enumerate(self.data_loader[dataloader_type]):
                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))
                total_num += y.size(0)
                _, soft_logit = self.IBnet(x, self.num_avg)
                if self.num_avg == 1 :
                    # log_likelihood
                    log_likelihood -= F.cross_entropy(soft_logit, y, reduction='sum').detach()
                    # accuracy
                    prediction = soft_logit.max(1)[1]
                    correct += torch.eq(prediction,y).float().sum().detach()
                elif self.num_avg > 1:
                    # log_likelihood
                    log_likelihood -= sum(F.cross_entropy(soft_logit[j,:,:], y, reduction='sum').detach() for j in range(self.num_avg))
                    # accuracy
                    predictions = [soft_logit[j,:,:].max(1)[1] for j in range(self.num_avg)]
                    correct += sum(torch.eq(predictions[j],y).float().sum().detach() for j in range(self.num_avg))
                # terminate if processed more than sample_size
                if idx*self.batch_size + 1 > sample_size: break
            # compute accuracy
            accuracy = correct/total_num/self.num_avg
            accuracies.append(accuracy)
            # compute average log_likelihood
            log_likelihood_av = log_likelihood/total_num/self.num_avg
            log_likelihoods_av.append(log_likelihood_av)

        # compute confidence intervals
        accuracy_confidence_intervals = st.norm.interval(alpha=confidence, loc=np.mean(torch.asarray(accuracies).cpu().numpy()), scale=st.sem(torch.asarray(accuracies).cpu().numpy()))
        log_likelihood_confidence_intervals = st.norm.interval(alpha=confidence, loc=np.mean(torch.asarray(log_likelihoods_av).cpu().numpy()), scale=st.sem(torch.asarray(log_likelihoods_av).cpu().numpy()))
        
        # output  
        self.accuracies_bootstrap = torch.asarray(accuracies)
        self.accuracy_confidence_intervals = accuracy_confidence_intervals
        self.log_likelihoods_bootstrap = torch.asarray(log_likelihoods_av)
        self.log_likelihood_confidence_intervals = log_likelihood_confidence_intervals