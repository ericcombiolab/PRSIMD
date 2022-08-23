#!/usr/bin/python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import time
from sklearn.metrics import roc_auc_score
import os
import utilities as util
import re

# model for the non-genetic factors
class log_linear_model(nn.Module):
    def __init__(self, n_factors, disease, n_cols_factor):
        super(log_linear_model, self).__init__()

        self.activation = torch.tanh
        self.disease = disease
        self.n_factors = n_factors
        self.n_cols_factor = n_cols_factor
        
        n_cols = np.sum(self.n_cols_factor)       
        self.W = nn.Parameter(torch.FloatTensor(torch.randn(2, n_cols)))       
        self.Gamma = nn.Parameter(torch.FloatTensor(torch.randn(2, n_factors+1))) 
    
                             
    def forward(self,f_data):
        f_data = f_data.unsqueeze(1)
        f_data = f_data.repeat(1,2,1)
        eleW_product = self.W * f_data # (n_classes, n_cols) * (batch_size, n_classes, n_cols) 
        
        start = 0     
        phi_collect = []       
        for i in self.n_cols_factor:
            
            one_factor = eleW_product[:,:, start: start+i]
           
            if one_factor.shape[2] > 1:
                one_factor = torch.sum(one_factor, dim=2)
            else:
                one_factor = one_factor.squeeze(2)
            phi_collect.append(one_factor)
            
            start += i
        
        
        phi = torch.stack(phi_collect,dim=2)  
        phi = self.activation(phi)
        phi_bias = torch.ones(phi.shape[0],phi.shape[1],1)
        phi = torch.cat((phi, phi_bias),dim=2)

        Gamma_times_Phi = self.Gamma * phi
        
        temp = torch.sum(Gamma_times_Phi, dim=2)
        temp = torch.exp(temp)
        temp_sum = torch.sum(temp, dim=1)
        temp_sum = temp_sum.unsqueeze(1)    
        temp_sum = temp_sum.repeat(1,2)     
        logit = temp / temp_sum
        return logit
        
        
        # phi_collect = self.encoding_functions(f_data) 
        # # logit
        # phi = torch.stack(phi_collect,dim=1)
        # phi = self.activation(phi)
        # phi = phi.permute(2,0,1)    # batch first
        # Gamma_times_Phi = []
        # for i in range(phi.shape[0]):
            # product = phi[i] * self.Gamma
            # Gamma_times_Phi.append(product)
        # Gamma_times_Phi = torch.stack(Gamma_times_Phi, dim=0) # (n_batch, n_classes, n_factors)
       
        # temp = torch.sum(Gamma_times_Phi, dim=2)
        # temp = torch.exp(temp)
        # temp_sum = torch.sum(temp, dim=1)
        # temp_sum = temp_sum.unsqueeze(1)    
        # temp_sum = temp_sum.repeat(1,2)     
        # logit = temp / temp_sum 
        # return logit
        


        
        
       
# model for the genetic factor
class logstic_regression_model(nn.Module):
    def __init__(self):
        super(logstic_regression_model, self).__init__()
        
        # self.w = nn.Parameter(torch.FloatTensor(torch.randn(2,1))) 
        # self.b = nn.Parameter(torch.FloatTensor(torch.randn(2,1))) 
        
        self.w = nn.Parameter(torch.FloatTensor(torch.randn(1,1))) 
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(1,1))) 
        
    def forward(self, score):
        logit_out = 1/(1+torch.exp(-(self.w * score + self.b))) 
        ne_logit_out = 1-logit_out
        logit = torch.cat((ne_logit_out,logit_out),dim=0)
        return logit    
        
        # temp_sum = torch.sum(logit_out, dim=0)
        # temp_sum = temp_sum.unsqueeze(0)    
        # temp_sum = temp_sum.repeat(2,1)  
        # logit = logit_out / temp_sum        
        # return logit
        
  
        
        
        
        

# model training 
def train_model(score, non_genetic_data, label, num_factors, model_save, loss_save, PRS_method, disease, alpha=1,beta=1,delta=1,batch_size=100,n_epoch=1000,n_cols_factor=None):    
   
    #### train model
    # model instances
    non_genetic = log_linear_model(num_factors, disease, n_cols_factor)
    genetic = logstic_regression_model()

    # optimizer construction
    #optimizer_loglinear = torch.optim.SGD(non_genetic.parameters(), lr=0.001, momentum=0.9, weight_decay=0.002)
    #optimizer_logisticRegression = torch.optim.SGD(genetic.parameters(), lr=0.001, momentum=0.9, weight_decay=0.002)
    #optimizer_loglinear = torch.optim.Adadelta(non_genetic.parameters(), rho = 0.001, weight_decay = 0.001)
    #optimizer_logisticRegression = torch.optim.Adadelta(genetic.parameters(), rho = 0.001, weight_decay = 0.001)
    optimizer_loglinear = torch.optim.Adam(non_genetic.parameters(), weight_decay = 0.001,amsgrad=True)
    optimizer_logisticRegression = torch.optim.Adam(genetic.parameters(), weight_decay = 0.001,amsgrad=True)
    
    n_batches = int(np.ceil(float(len(label['train'])) / float(batch_size)))
    
    val_loss_best = 100000
    val_loss_epoch = []
    train_loss_epoch = []

    start_time = time.time()    

    for epoch in range(n_epoch):
    #for epoch in range(2): # for debug
        loss_batch = []
        batches = random.sample(range(n_batches), n_batches) # suff batch index 
        for i in batches:
            ## mini-batch data
            batch_y = label['train'][batch_size * i : batch_size * (i + 1)]
            batch_data = non_genetic_data['train'][batch_size * i : batch_size * (i + 1)]
            batch_PRS= score['train'][batch_size * i : batch_size * (i + 1)]
       
            ## forward
            optimizer_loglinear.zero_grad() 
            optimizer_logisticRegression.zero_grad()
            
            logistic_re_results = genetic(batch_PRS).T # (n_batch, n_classes)      
            log_linear_results = non_genetic(batch_data) # (n_batch, n_classes)
 
            ## loss & backward 
            logistic_re_loss = F.cross_entropy(logistic_re_results, batch_y)
            log_linear_loss = F.cross_entropy(log_linear_results, batch_y)
            kl_loss = util.kl_loss(log_linear_results, logistic_re_results)  
            #kl_loss = util.kl_loss(logistic_re_results, log_linear_results)
            loss =  logistic_re_loss + alpha * kl_loss + beta * log_linear_loss
            loss.backward()
            
            ## update
            optimizer_loglinear.step()
            optimizer_logisticRegression.step()
            
            ## loss information saving (each batch)
            loss_batch.append(loss.data) 
            
        ## loss information saving (each epoch)
        train_loss_epoch.append(np.mean(loss_batch)) 

        if np.mod(epoch, 50) == 0:
            count_time = time.time()-start_time
            print("run epoch:%d , time consumed: %d s" % (epoch,int(count_time) ))
        
        ## validation for model selection
        val_loss = validation_loss(genetic, non_genetic, score, non_genetic_data, label, alpha, beta, batch_size)    
        val_loss_epoch.append(val_loss.detach().numpy())
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            state = {'non_genetic':non_genetic.state_dict(), 'optimizer_non_genetic':optimizer_loglinear.state_dict(),
                'genetic':genetic.state_dict(), 'optimizer_genetic':optimizer_logisticRegression.state_dict(), 'epoch':epoch,'alpha':alpha,'beta':beta}         
            torch.save(obj=state, f= os.path.join(model_save, 'model_'+ PRS_method + '_' + disease + '_' + str(alpha)+'_'+str(beta)+'.pth'))
     
    ## training information saving
    np.save(os.path.join(loss_save, 'train_loss_'+ PRS_method + '_' + disease + '_'+str(alpha)+'_'+str(beta)), train_loss_epoch)
    np.save(os.path.join(loss_save, 'val_loss_'+ PRS_method + '_' + disease + '_'+str(alpha)+'_'+str(beta)), val_loss_epoch)

    
def validation_loss(logstic_model, loglinear_model, score ,non_genetic_data, label, alpha, beta, batch_size):
    # # validation prediction
    val_result_non_genetic = loglinear_model(non_genetic_data['validation'])
    val_result_genetic = logstic_model(score['validation']).T 
    # validation loss  
    val_loss_non_genetic = F.cross_entropy(val_result_non_genetic, label['validation'])
    val_loss_genetic = F.cross_entropy(val_result_genetic, label['validation'])
    val_loss_kl = util.kl_loss(val_result_non_genetic, val_result_genetic)  
    #val_loss_kl = util.kl_loss(val_result_genetic, val_result_non_genetic)  
    val_loss =  val_loss_genetic + alpha * val_loss_kl + beta * val_loss_non_genetic  
    return val_loss
    
    
    # n_batches = int(np.ceil(float(len(label['validation'])) / float(batch_size)))
    
    # loss_batch = []
    # batches = random.sample(range(n_batches), n_batches) # suff batch index 
    # for i in batches:
        # ## mini-batch data
        # batch_y = label['validation'][batch_size * i : batch_size * (i + 1)]
        # batch_data = non_genetic_data['validation'][batch_size * i : batch_size * (i + 1)]
        # batch_PRS= score['validation'][batch_size * i : batch_size * (i + 1)]

        # val_result_non_genetic = loglinear_model(batch_data)
        # val_result_genetic = logstic_model(batch_PRS).T 
        # # validation loss  
        # val_loss_non_genetic = F.cross_entropy(val_result_non_genetic, batch_y)
        # val_loss_genetic = F.cross_entropy(val_result_genetic, batch_y)
        # val_loss_kl = util.kl_loss(val_result_non_genetic, val_result_genetic)  
        # val_loss = delta* val_loss_genetic + alpha * val_loss_kl + beta * val_loss_non_genetic
        # loss_batch.append(val_loss)
    # loss = torch.mean(torch.Tensor(loss_batch))
    # return loss


def validate_best_alpha_beta(model_save, PRS_method, disease, score, non_genetic_data, label, alpha, beta,  num_factors, n_cols_factor):
    collect = []
    for i in range(len(alpha)):
        collect_mid = []
        for j in range(len(beta)):
            #collect_mid_mid = []
            #for k in range(len(delta)):
            val_non_genetic = log_linear_model(num_factors, disease, n_cols_factor)
            val_genetic = logstic_regression_model()
            
            best_model = os.path.join(model_save, 'model_'+ PRS_method + '_' + disease + '_' + str(alpha[i])+'_'+str(beta[j]) +'.pth')
            reload_states = torch.load(best_model)          
            val_non_genetic.load_state_dict(reload_states['non_genetic'])
            val_genetic.load_state_dict(reload_states['genetic'])
            
            PRSIMD_pred = val_genetic(score['validation']).T + val_non_genetic(non_genetic_data['validation'])
            y_pred_class = torch.argmax(PRSIMD_pred,dim=1)
            y_pred_score = PRSIMD_pred[:,1]
            
            #y_pred_class  =  y_pred_class.detach().numpy()
            y_pred_score  =  y_pred_score.detach().numpy()
            y = label['validation'].detach().numpy() 
              
            #AUROC = roc_auc_score(y, y_pred_class) 
            AUROC = roc_auc_score(y, y_pred_score)       
            #collect_mid_mid.append(AUROC)
            collect_mid.append(AUROC)
        collect.append(collect_mid)
        
    AUROCs_gridMatrix= np.array(collect)
    idx_best_hyper = np.argwhere(AUROCs_gridMatrix == np.max(AUROCs_gridMatrix))[0]    # row- alpha, col- beta        
    return AUROCs_gridMatrix, idx_best_hyper
    
    
# model test   
def test_model(model_save, PRS_method, disease, score, non_genetic_data, label, alpha, beta,  num_factors, n_cols_factor):    
    test_non_genetic = log_linear_model(num_factors, disease, n_cols_factor)
    test_genetic = logstic_regression_model()
    
    best_model = os.path.join(model_save, 'model_'+ PRS_method + '_' + disease + '_' + str(alpha)+'_'+str(beta)+'.pth')
    reload_states = torch.load(best_model)   
    test_non_genetic.load_state_dict(reload_states['non_genetic'])
    test_genetic.load_state_dict(reload_states['genetic'])
    
    PRSIMD_pred = test_genetic(score['test']).T + test_non_genetic(non_genetic_data['test'])
    #y_pred_class = torch.argmax(PRSIMD_pred,dim=1)
    y_pred_score = PRSIMD_pred[:,1]
    
    #y_pred_class  =  y_pred_class.detach().numpy()
    y_pred_score  =  y_pred_score.detach().numpy()
    y = label['test'].detach().numpy() 
     
    AUROC = roc_auc_score(y, y_pred_score)
    return AUROC, y_pred_score, y



def loding_data(PRS_score_path, PRS_Method, Disease,non_genetic_data_path, label_path, category_list):
    score_dict = {}
    non_genetic_data_dict = {}
    label_dict = {}
    valt = {'train':'train','validation':'val', 'test':'test'}
    for _set in ['train','validation','test']:
        score, _ = util.readbyLines(os.path.join(PRS_score_path, _set + '_score_' + PRS_Method + '_' + Disease + '.txt'),datatype="float")
        score_dict[_set] =  Variable(torch.FloatTensor(score))
        
        collect = []
        for categ in category_list:
            collect.append( pd.read_table(os.path.join(non_genetic_data_path, 'data_mat', Disease.lower(), categ + '_' + valt[_set] + '_data.txt')))
            data = pd.concat(collect, axis=1)
            data['Age when attended assessment centre'] /= 10 # used age group
            data['Age when attended assessment centre'] = data['Age when attended assessment centre'].astype('int')
        #non_genetic_data_dict[_set] = Variable(torch.FloatTensor(data))
        non_genetic_data_dict[_set] = data
      
        y = pd.read_table(os.path.join(label_path, Disease.lower()+'_' + valt[_set] + '_y.txt'),header=None).values.ravel()      
        label_dict[_set] = Variable(torch.LongTensor(y.astype('long'))) 
    
    return score_dict, non_genetic_data_dict, label_dict



def get_data_info_for_model_construction(non_genetic_data, Disease, non_genetic_data_path, category_list):    
    collect = []
    for cate in category_list:
        collect.append(pd.read_table(os.path.join(non_genetic_data_path, Disease.lower()+'_'+cate+'.txt')))
    df_factors_infor = pd.concat(collect,axis=0)
    n_factors = df_factors_infor.shape[0]
    
    # get the info for model initialization
    col_name = non_genetic_data['train'].columns.values
    n_cols_factor = [] # the number of columns that each factor data accounted
    count = 0
    for name in df_factors_infor['field_name'].values:  
        for col in col_name:
            tmp = col.split(':')   
            if name == tmp[0]:
                count+=1
      
        n_cols_factor.append(count)
        count = 0
                   
    return n_factors, n_cols_factor
    


def creat_result_folders(save_dir):   
    if save_dir is not None:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
                     
    model_dir = os.path.join(save_dir, 'model_save')
    if model_dir is not None:
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
            
    loss_dir = os.path.join(save_dir, 'loss_save')
    if loss_dir is not None:
        if not os.path.isdir(loss_dir):
            os.mkdir(loss_dir)

    return model_dir, loss_dir




if __name__ == '__main__':
    torch.set_num_threads(10)
  
    ### user input params
    PRS_score_path = './PRS'
    non_genetic_data_path = './non_genetic_data'
    label_path = './eid_label'
    category_list =['primary_demographics','lifestyle','physical_measures']#'physical_measures' 'lifestyle'
    Disease = 'CAD'
    PRS_Method = 'LDpred2'# DBSLMM, SBLUP, PRSice2, P+T, LDpred2
    save_dir = './PRSIMD_results_MRfactors'

   
    alpha_grid = [1,0.1,0.01,0.001] 
    beta_grid = [1,0.1,0.01,0.001]   
     

    n_epoch = 300
    batch_size = 128
    
    ### data loading
    score, non_genetic_data, label = loding_data(PRS_score_path, PRS_Method, Disease, non_genetic_data_path, label_path, category_list) 
    n_factors, n_cols_factor= get_data_info_for_model_construction(non_genetic_data, Disease, non_genetic_data_path, category_list)
    print('number of the factors:\t', n_factors)

    Data_input= {}
    for i in ['train','test','validation']:
        Data_input[i] = Variable(torch.FloatTensor(non_genetic_data[i].values)) 

    
    model_dir, loss_dir = creat_result_folders(save_dir)

    ### training
    for i in range(len(alpha_grid)):
        for j in range(len(beta_grid)):
            #for k in range(len(delta_grid)):
                # train model
            print('\n')
            print('alpha:\t', alpha_grid[i])
            print('beta:\t', beta_grid[j])
            #print('delta:\t', delta_grid[k])

            train_model(score, Data_input, label, num_factors=n_factors, model_save=model_dir, loss_save=loss_dir, PRS_method=PRS_Method,
                    disease=Disease, alpha=alpha_grid[i], beta=beta_grid[j],batch_size=batch_size, n_epoch=n_epoch, n_cols_factor=n_cols_factor)              
                    
    ### grid search
    _, best_hyper_idx = validate_best_alpha_beta(model_save=model_dir, PRS_method=PRS_Method, disease=Disease, score=score, non_genetic_data=Data_input,
                    label=label, alpha=alpha_grid, beta=beta_grid, num_factors=n_factors, n_cols_factor=n_cols_factor)
                    
    ## test                 
    test_auroc, y_pred, y_test = test_model(model_save=model_dir, PRS_method=PRS_Method, disease=Disease, score=score, non_genetic_data=Data_input, 
                    label=label, alpha=alpha_grid[best_hyper_idx[0]], beta=beta_grid[best_hyper_idx[1]],  num_factors=n_factors, n_cols_factor=n_cols_factor)   
    
    util.save_txt(os.path.join(save_dir, Disease+'_'+PRS_Method+'_y_pred.txt'), y_pred)
    util.save_txt(os.path.join(save_dir, Disease+'_'+PRS_Method+'_y_test.txt'), y_test)
    util.save_txt(os.path.join(save_dir, Disease+'_'+PRS_Method+'_best_hyper.txt'), best_hyper_idx)
    util.save_txt(os.path.join(save_dir, Disease+'_'+PRS_Method+'_auroc.txt'), [test_auroc])
      
    print(Disease,'\t',PRS_Method,'\t',test_auroc)
    
    # for i in range(len(alpha_grid)):
        # for j in range(len(beta_grid)):
            # test_auroc, y_pred, y_test = test_model(model_save=model_dir, PRS_method=PRS_Method, disease=Disease, score=score, non_genetic_data=Data_input, 
                            # label=label, alpha=alpha_grid[i], beta=beta_grid[j], num_factors=n_factors, n_cols_factor=n_cols_factor)   
            
            # print(test_auroc)
