import numpy as np
import pandas as pd
import random
import time
from sklearn.metrics import roc_auc_score
import os
import utilities as util
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def loding_data(PRS_score_path, PRS_Method, Disease,non_genetic_data_path, label_path, category_list):
    score_dict = {}
    non_genetic_data_dict = {}
    label_dict = {}
    valt = {'train':'train','validation':'val', 'test':'test'}
    for _set in ['train','validation','test']:
        score, _ = util.readbyLines(os.path.join(PRS_score_path, _set + '_score_' + PRS_Method + '_' + Disease + '.txt'),datatype="float")
        score_dict[_set] =  score
        
        collect = []
        for categ in category_list:
            collect.append( pd.read_table(os.path.join(non_genetic_data_path, 'data_mat', Disease.lower(), categ + '_' + valt[_set] + '_data.txt')))
            data = pd.concat(collect, axis=1)
            #data['Age when attended assessment centre'] /= 10 # used age group
            #data['Age when attended assessment centre'] = data['Age when attended assessment centre'].astype('int')
        #non_genetic_data_dict[_set] = Variable(torch.FloatTensor(data))
        non_genetic_data_dict[_set] = data
      
        y = pd.read_table(os.path.join(label_path, Disease.lower()+'_' + valt[_set] + '_y.txt'),header=None).values.ravel()      
        label_dict[_set] = y.astype('long')
    
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
    d_type = []
    d_name = []
    count = 0
    for name in df_factors_infor['field_name'].values:   # get the columns father field 
        for col in col_name:
            tmp = col.split(':')   
            if name == tmp[0]:
                count+=1
      
        n_cols_factor.append(count)
        count = 0
    
        d_type.append(df_factors_infor[df_factors_infor['field_name']==name]['data_type'].values[0])
        d_name.append(name)
        
    return n_factors, n_cols_factor, d_type, d_name
    
  
def _continuous_riskbin_construction(values, label):
    np.seterr(divide='ignore', invalid='ignore')
    
    hist, edges = np.histogram(np.array(values), bins=100)
    idx_case = np.argwhere(label==1).ravel()
    case_values = np.array(values)[idx_case]
    
    n_each_bin_list = []
    n_each_bin_list_ = []
    for i in range(len(edges)-1):
        tmp_1 = np.where(values >= edges[i])[0]  
        tmp_1_ = np.where(case_values >= edges[i])[0]  
      
        if i== (len(edges)-2) :
            tmp_2 = np.where(values <= edges[i+1])[0]
            tmp_2_ = np.where(case_values <= edges[i+1])[0]
        else:
            tmp_2 = np.where(values < edges[i+1])[0]   
            tmp_2_ = np.where(case_values < edges[i+1])[0]
        tmp = list(set(tmp_1) & set(tmp_2))
        tmp_ = list(set(tmp_1_) & set(tmp_2_))
   
        n_each_bin_list.append(len(tmp))
        n_each_bin_list_.append(len(tmp_))
    
    risk_bins = np.array(n_each_bin_list_) / np.array(n_each_bin_list)
    risk_bins = np.nan_to_num(risk_bins)
    return risk_bins, edges    


def _transform_into_riskProb_by_bin(values, risk_bins, edges):
    idx_list = []
    for v in values:
        for i in range(len(edges)):
            if v >= edges[-1]:
                idx = len(edges)-2 
                idx_list.append(idx)
                break
            elif not v >= edges[i]:           
                idx = (i-1) 
                idx_list.append(idx)
                break
                
    risk_prob = risk_bins[idx_list]
    return risk_prob


def _dummy_reverse(dummy):
    ## reverse dummy format to categorical format (represented by specific values) 
    first = np.zeros(dummy.shape[0]).astype(type(dummy)).reshape(-1,1) # add first col  
    dummy_recov = np.concatenate((first, dummy), axis=1) 
    for j in range(dummy_recov.shape[0]):
        if not dummy_recov[j].any() != 0:
            dummy_recov[j, 0] = 1         
    recov = np.argmax(dummy_recov, axis = 1)
    return recov    
    
    
def _categorical_riskbin_construction(dummy, dummy_recov, label):
    idx_case = np.argwhere(label==1).ravel()
    case_recov = dummy_recov[idx_case]
    tmp1_ = []
    tmp2_ = []
    for this_bin in range(dummy.shape[1] +1 ):
        tmp1 = np.argwhere(dummy_recov == this_bin).shape[0]
        tmp2 = np.argwhere(case_recov == this_bin).shape[0]
        tmp1_.append(tmp1)
        tmp2_.append(tmp2)
    
    tmp = np.array(tmp2_)/np.array(tmp1_)
    return tmp   


def _transform_into_riskProb_by_bin_cate(data, risk_bins):
    dummy = data
    recov = _dummy_reverse(dummy)
    prob = risk_bins[recov]
    return prob


def _integer_riskbin_construction(values, label):
    idx_case = np.argwhere(label==1).ravel()
    case_recov = values[idx_case]
    n = list(set(values))
    
    tmp1_ = []
    tmp2_ = []
    for this_bin in n:
        tmp1 = np.argwhere(values == this_bin).shape[0]
        tmp2 = np.argwhere(case_recov == this_bin).shape[0]
        tmp1_.append(tmp1)
        tmp2_.append(tmp2)
    
    tmp = np.array(tmp2_)/np.array(tmp1_)
    return tmp, n   



def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]



def _transform_into_riskProb_by_bin_int(data, risk_bins, bin_integers):
    # transform into index
    idx_collect = []
    for k in data:      
        idx = np.argwhere(np.array(bin_integers) == k)
        if idx.size > 0:
            idx_collect.append(idx[0][0])                  
        else:
            tmp = find_nearest(np.array(bin_integers),k)
            idx = np.argwhere(np.array(bin_integers) == tmp)[0][0]
            idx_collect.append(idx)
    # transform into probablity
    risk_prob = risk_bins[idx_collect]
  
    return risk_prob



class CRS(nn.Module):
    def __init__(self, n_factors):
        super(CRS, self).__init__()
        self.weights_raw = nn.Parameter(torch.FloatTensor(torch.randn(n_factors)))    

    def forward(self, data):
        weights = F.softmax(self.weights_raw, dim=0)
        weighted = data * weights
        crs = torch.sum(weighted, dim=1)
        return crs
       
        
    
if __name__ == '__main__':
 
    ### user input params
    PRS_score_path = './PRS'
    non_genetic_data_path = './non_genetic_data'
    label_path = '/home/comp/csyuxu/PRSIMD/code/eid_label'
    category_list =['primary_demographics','lifestyle','physical_measures']#'physical_measures' 'lifestyle'
    Disease = 'CAD'
    PRS_Method = 'PRSice2'# DBSLMM, SBLUP, PRSice2, P+T, LDpred2
    save_dir = './CRS_results_MRfactors'
    
    ### data loading
    score, non_genetic_data, label = loding_data(PRS_score_path, PRS_Method, Disease, non_genetic_data_path, label_path, category_list) 
    n_factors, n_cols_factor, datatypes, fieldnames= get_data_info_for_model_construction(non_genetic_data, Disease, non_genetic_data_path, category_list)
    print('number of the factors:\t', n_factors)

    Data_input= {}
    for i in ['train','test','validation']:
        Data_input[i] = non_genetic_data[i].values



    ### binning PRS and transform it into risk probablity 
    risk_bins, edges = _continuous_riskbin_construction(score['train'], label['train'])
    PRS_trans = {}
    for i in ['train','test','validation']:
        PRS_trans[i] = _transform_into_riskProb_by_bin(score[i], risk_bins, edges)
    
    
    ### binning non-genetic factors   
    current_col = 0
    collect_dict = {'train':[],'test':[],'validation':[]}
    for i in range(len(n_cols_factor)):
        if datatypes[i] == 'Integer':
            risk_bins, bin_integers = _integer_riskbin_construction(Data_input['train'][:,current_col], label['train'])
            for j in ['train','test','validation']:
                collect_dict[j].append(_transform_into_riskProb_by_bin_int(Data_input[j][:,current_col], risk_bins, bin_integers))      

        if datatypes[i] == 'Categorical (single)' or datatypes[i] == 'Categorical (multiple)':  
            dummy = Data_input['train'][:,current_col: current_col + n_cols_factor[i]] 
            recov = _dummy_reverse(dummy) # recovery from dummy factors
            risk_bins =  _categorical_riskbin_construction(dummy, recov, label['train'])
            for j in ['train','test','validation']:
                collect_dict[j].append(_transform_into_riskProb_by_bin_cate(Data_input[j][:,current_col: current_col + n_cols_factor[i]], risk_bins))      

        if datatypes[i] == 'Continuous':         
            risk_bins, edges = _continuous_riskbin_construction(Data_input['train'][:,current_col], label['train'])
            for j in ['train','test','validation']:
                collect_dict[j].append(_transform_into_riskProb_by_bin(Data_input[j][:,current_col], risk_bins, edges))      
          
        current_col += n_cols_factor[i]
    
    # risk probablity data format
    Data_input_with_PRS = {}
    for j in ['train','test','validation']:
        Data_input_with_PRS[j] =  Variable(torch.FloatTensor( np.concatenate( (np.array(collect_dict[j]).T, PRS_trans[j].reshape(-1,1)), axis=1 ) ))


    ### saving path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_save = os.path.join(save_dir, 'model')
    if not os.path.exists(model_save):
        os.mkdir(model_save)
        
        
    ### train
    batch_size = 128
    n_batches = int(np.ceil(float(len(label['train'])) / float(batch_size)))
    model = CRS(Data_input_with_PRS['train'].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = 0.001,amsgrad=False)
    
    val_loss_best = 100000
    n_epoch = 1000
    loss_batch=[]  
    train_loss_epoch=[]    
    for epoch in range(n_epoch):
        batches = random.sample(range(n_batches), n_batches) # suff batch index 
        for i in batches:            
            
            batch_y = Variable(torch.FloatTensor( label['train'][batch_size * i : batch_size * (i + 1)] ))
            batch_data = Data_input_with_PRS['train'][batch_size * i : batch_size * (i + 1)]


            optimizer.zero_grad() 
            CRS_score = model(batch_data)
            loss = F.binary_cross_entropy(CRS_score, batch_y)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.data) 
        train_loss_epoch.append(np.mean(loss_batch)) 
        
        val_loss = F.binary_cross_entropy(model(Data_input_with_PRS['validation']), Variable(torch.FloatTensor(label['validation'])) ) 
        
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            state = {'CRS_model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}         
            torch.save(obj=state, f= os.path.join(model_save, 'model_' + PRS_Method + '_' + Disease + '_' + '.pth'))
    
    ### test 
    model_test = CRS(Data_input_with_PRS['test'].shape[1])
    
    reload_states = torch.load(os.path.join(model_save, 'model_' + PRS_Method + '_' + Disease + '_' + '.pth'))          
    model_test.load_state_dict(reload_states['CRS_model'])
    
    test_score = model_test(Data_input_with_PRS['test'])
    
    y_pred_score  =  test_score.detach().numpy()
    y = label['test']
       
    AUROC = roc_auc_score(y, y_pred_score)
        
    util.save_txt(os.path.join(save_dir, 'CRS_' + Disease+'_'+PRS_Method+'_y_pred.txt'), y_pred_score)
    util.save_txt(os.path.join(save_dir, 'CRS_' + Disease+'_'+PRS_Method+'_y_test.txt'), y)
    util.save_txt(os.path.join(save_dir, 'CRS_' + Disease+'_'+PRS_Method+'_auroc.txt'), [AUROC])
        
    print(AUROC)
    