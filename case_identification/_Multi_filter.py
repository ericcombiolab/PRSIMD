import pandas as pd
import numpy as np
import os
from collections import defaultdict



def instances_combination(colData_dir, factor):
    cols_idx = [i for i in range(factor['col_start'],factor['col_end']+1)]
    collect = []
    for j in cols_idx:
        collect.append( np.load(os.path.join(colData_dir, str(j)+'.npy'))[1:] )
    
    if factor['data_type'] != 'Categorical (multiple)':       
        _data = collect[0]
        for k in range(1, len(collect)):
            current_ins = collect[k]            
            for v in range(len(collect[k])):
                if _data[v] == '' and current_ins[v] != '': # missing value in a instances
                    _data[v] = current_ins[v]
                    
            # ## debug, fill missings by data from the other instance
            # idxx = np.argwhere(_data=='').ravel()
            # print(len(idxx))
        return _data   
    else:
        _data = np.array(collect).T # rows: participants; cols: instances 
        return _data
    
    
    
if __name__ == '__main__':   
    
    n_participants = 502506
    collect = []
    colData_dir = '/tmp/csyuxu/ukbb/data_cols'
    eid_column_path = '/home/comp/ericluzhang/xuyu/eid/0.npy'
    save_dir = './Multi_filter_results'
    
    
    ## eid <-> index in the dataset
    ind_eid = np.load(eid_column_path)[1:]

    ## data-field info
    df = pd.read_table('pca_dcode_opcs_ethic_info.txt')
    
    ## self-reported code
    cancer_self = {'breast cancer':['1002']}
    noncancer_self = {'Coronary artery disease':['1075'], 'Inflammatory bowel disease':['1461','1462','1463'], 'Type2 diabetes':['1223']}
    
    ## OPCS-4 code
    opcs = {'Coronary artery disease': ['K40'+str(i) for i in range(1,5)] + ['K41'+str(i) for i in range(1,5)] + ['K45'+str(i) for i in range(1,6)] 
                                + ['K491','K492','K498','K499','K758','K759','K502'] + ['K75'+str(i) for i in range(1,5)] }
    
    ## saving path
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    for factor in df.iterrows(): 
        factor = factor[1] 
        
        ## load data
        if factor['col_start'] != factor['col_end']:    # having multi-instance
            data = instances_combination(colData_dir, factor)
        else:                                           # single instance
            data = np.load(os.path.join(colData_dir, str(factor['col_start'])+'.npy'))[1:]
       

        #### filter participants with the different conditions ####
        
        ## participants used in PCA
        if factor['field_name'] == 'Used in genetic principal components': 
            idx = []
            for i in range(n_participants-1):
                if data[i] == '1': 
                    idx.append(i)
                    
                # # debug   
                # if i>10:                   
                    # print(idx,ind_eid[idx])
                    # break
            eid_filtered = ind_eid[idx]
            f = open(os.path.join(save_dir, factor['field_name']+'_eid.txt'),'w')
            for eid in eid_filtered:
                f.write(str(eid)+'\n')
            f.close()
          
        ## self-reported cancer 
        if factor['field_name'] == 'Cancer code, self-reported':      
       
            for disease in cancer_self.keys():
                print('Disease:\t', disease)
                
                idx_collect = []
                for i in range(n_participants-1):  
                    for code in cancer_self[disease]:                
                        if code in data[i]:                     
                            idx_collect.append(i)
                    
                    # debug   
                    # if i>100:                                          
                        # break
                
                eid_filtered = ind_eid[idx_collect]   # transform into eid 
                
                f = open(os.path.join(save_dir, disease +'_selfreport_eid.txt'),'w')
                for eid in eid_filtered:
                    f.write(str(eid)+'\n')
                f.close()
          
        ## self-reported ill non-cancer     
        if factor['field_name'] == 'Non-cancer illness code, self-reported':       
            for disease in noncancer_self.keys():
                print('Disease:\t', disease)
                
                idx_collect = []
                for i in range(n_participants-1):
                    for code in noncancer_self[disease]:
                        if code in data[i]:                     
                            idx_collect.append(i)
                            break
                    #debug   
                    # if i>1000:                                          
                        # break
                
                eid_filtered = ind_eid[idx_collect]   # transform into eid 
                
                f = open(os.path.join(save_dir, disease +'_selfreport_eid.txt'),'w')
                for eid in eid_filtered:
                    f.write(str(eid)+'\n')
                f.close()
          
        ## OPCS-4 code
        if factor['field_name'] == 'Operative procedures - main OPCS4':       
            for disease in opcs.keys():
                print('Disease:\t', disease)
                
                idx_collect = []
                for i in range(n_participants-1):
                    for code in opcs[disease]:
                        if code in data[i]:                     
                            idx_collect.append(i)
                            break
                    # #debug   
                    # if i>1000:                                          
                        # break
                
                eid_filtered = ind_eid[idx_collect]   # transform into eid 
                
                f = open(os.path.join(save_dir, disease +'_opcs4_eid.txt'),'w')
                for eid in eid_filtered:
                    f.write(str(eid)+'\n')
                f.close()
       
        ## Ethnic 
        if factor['field_name'] == 'Ethnic background':
            idx = []
            for i in range(n_participants-1):
                if data[i] == '1001': 
                    idx.append(i)
                    
                # # debug   
                # if i>10:                   
                    # print(idx,ind_eid[idx])
                    # break
            eid_filtered = ind_eid[idx]
            f = open(os.path.join(save_dir, 'white_british_eid.txt'),'w')
            for eid in eid_filtered:
                f.write(str(eid)+'\n')
            f.close()
        
            
            
