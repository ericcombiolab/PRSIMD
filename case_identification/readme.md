# The scripts used to identify the disease cases from UK biobank

### ICD-code 
`_HES_ICDcode_filter.py` is used to match ICD-code in HES record. 

```
    hes_diag_path  = '/home/comp/ericluzhang/UKBB/HES/hesin_diag.txt'
    #eid_column_path = '/home/comp/ericluzhang/xuyu/eid/0.npy'
    save_dir = './HES_fiter_result'
    
    disease_icd_10 = {'Type2 diabetes': [r'E11.*'], 'Coronary artery disease':[r'I21.*',r'I22.*',r'I23.*',r'I241',r'I252'],
            'Breast cancer':[r'C50.*'], 'Inflammatory bowel disease':[r'K51.*']}
    disease_icd_9 = {'Type2 diabetes': [], 'Coronary artery disease':[r'410.*',r'4110',r'412.*',r'42979'],
            'Breast cancer':[r'174',r'1749'],'Inflammatory bowel disease':[r'555.*']}

```

* Input: HES diagnosis file from UK biobank; disease ICD-code version 9 or (and) 10
* Output: [disease_name.txt] contains the eids of participants diagnosed with that disease

### Self-reported code 
`_Multi_filter.py` 

```
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
```

* Input: The corresponding column data that were extracted from `ukbxxx.csv` dataset; disease code in self-report columns 
* Output: [disease_name_selfreport_eid.txt] contains the eids of participants reported that disease

> This script also can be used to screen the participants with specific OPCS-4 codes/ethnic backgroud.