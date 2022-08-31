#!/usr/bin/python
import re
from collections import defaultdict
import numpy as np
import os


def HES_diagnosis(hes_diag_path, eid_column_path, disease_icd_10, disease_icd_9, save_dir):

    n_participants = 502506


    hesdiag_file=open(hes_diag_path,'r')
    #ind_eid = np.load(eid_column_path)[1:]
    
    
    collect = defaultdict(list)

    count=0
    for line in hesdiag_file:

        ## debug only
        # if count == 1000:         
            # break
            
        ## progress rate
        count += 1
        if np.mod(count,1000) == 0:
           print('Iterated entries:',count) 
        
        
        ## HES diagnosis file columns
        A = line.strip('\n').split('\t') 
        eid = A[0]
        ins_index = A[1]
        level = A[3]
        ICD9 = A[4]
        ICD10 = A[6]
        
        ## ignore first row (columns name)
        # if eid!='eid':
            # index = int(np.where(ind_eid==eid)[0])   
         
        ## HES primary diagnosis
        if level == '1': 
            # for each disease
            for d in disease_icd_10.keys():  
                # icd-10
                # for each icd-code in a disease
                if disease_icd_10[d] != []:           
                    for icd in disease_icd_10[d]:
                        flag_match = re.match(icd, ICD10)
                        # return if matched one of the disease icd-code
                        if flag_match != None:
                            collect[d].append(eid)
                            break
                            
                # icd-9
                # for each icd-code in a disease
                if disease_icd_9[d] != []:           
                    for icd in disease_icd_9[d]:
                        flag_match = re.match(icd, ICD9)
                        # return if matched one of the disease icd-code
                        if flag_match != None:
                            collect[d].append(eid)
                            break
                          
    hesdiag_file.close()


    print('removing duplicated eid, saving to txt file')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
   
    
    for disease in collect.keys():
        #print(disease, collect[disease])
        eid_dup_removed = list(set(collect[disease]))
        eid_dup_removed.sort(key = collect[disease].index)
        
        f = open(os.path.join(save_dir, disease+'_HES_case_eid.txt'),'w')
        for case_eid in eid_dup_removed:       
            f.write(str(case_eid) + '\n')  
        f.close()
        
    
if __name__ == '__main__':

    hes_diag_path  = '/home/comp/ericluzhang/UKBB/HES/hesin_diag.txt'
    #eid_column_path = '/home/comp/ericluzhang/xuyu/eid/0.npy'
    save_dir = './HES_fiter_result'
    
    disease_icd_10 = {'Type2 diabetes': [r'E11.*'], 'Coronary artery disease':[r'I21.*',r'I22.*',r'I23.*',r'I241',r'I252'],
            'Breast cancer':[r'C50.*'], 'Inflammatory bowel disease':[r'K51.*']}
    disease_icd_9 = {'Type2 diabetes': [], 'Coronary artery disease':[r'410.*',r'4110',r'412.*',r'42979'],
            'Breast cancer':[r'174',r'1749'],'Inflammatory bowel disease':[r'555.*']}

    
    # filter diagnosis in HES_diag file 
    HES_diagnosis(hes_diag_path, eid_column_path=None, disease_icd_10, disease_icd_9, save_dir)
