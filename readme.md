# PRSIMD: A machine learning model for disease risk prediction by integrating genetic and non-genetic factors
Paper submitted to Biorxiv: https://doi.org/10.1101/2022.08.22.504882


**************

## UK biobank
+ [Data showcase](https://biobank.ndph.ox.ac.uk/showcase/)
+ [Genetic data](https://biobank.ndph.ox.ac.uk/ukb/ukb/docs/ukbgene_instruct.html)

**************

## Genome-wide association studies (GWAS) summary statistics
### The disease-susceptible SNPs for PRS calculation
+ Coronary artery disease: [Nikpay, Majid, et al.](https://www.ebi.ac.uk/gwas/studies/GCST003116)
+ Type 2 diabetes: [Scott, Robert A., et al.](http://diagram-consortium.org/downloads.html)

### The SNPs associated with the non-genetic factors
+ Coronary artery disease:  
The `exposure` columns of [ivw-cad.csv](https://github.com/yuxu-1/PRSIMD/blob/master/mr/cad/ivw-cad.csv) include the ID of the GWAS associated with that non-genetic factor. 
For example, `Arm fat mass (left)  || id:ukb-b-8338` indicates that the GWAS ID for the factor `Arm fat mass (left)` is `ukb-b-8338`. Retrieve this ID from [openGWAS](https://gwas.mrcieu.ac.uk/) database to get the information of GWAS summary statistics for `Arm fat mass (left)`.  
+ Type 2 diabetes: 
[ivw-t2d.csv](https://github.com/yuxu-1/PRSIMD/blob/master/mr/t2d/ivw-t2d.csv)

**************

## Code Environment
The file of conda environment installation is provided.
1. Download "environment.yml" file in this repository;
2. Modify the path in the final line of "environment.yaml" file according to user's conda path (e.g.,`"/home/comp/csyuxu/anaconda3/envs/PRSIMD"`);
3. Run this command to install all python dependency for PRSIMD
``` bash
conda env create -f environment.yml
```  
4. Activate conda env
``` bash
conda activate PRSIMD
```  
**************
## Identify disease cases from UK biobank 
[HES record, Self-reports](https://github.com/yuxu-1/PRSIMD/tree/master/case_identification)

**************
## UK biobank non-genetic data processing
[Link](./non_genetic_data)

**************

## Train and evaluate PRSIMD
#### Python script: PRSIMD.py
Run the blow script to reproduce the experimental results of PRSIMD (LDpred2) on CAD.
``` bash
python PRSIMD.py  
```

The key input files and settings:
```
    ### user input params
    PRS_score_path = './PRS'
    non_genetic_data_path = './non_genetic_data'
    label_path = './eid_label'
    category_list =['primary_demographics','lifestyle','physical_measures']#'physical_measures' 'lifestyle'
    Disease = 'CAD' # T2D
    PRS_Method = 'LDpred2'# DBSLMM, SBLUP, PRSice2, P+T, LDpred2
    save_dir = './PRSIMD_results_MRfactors'

```

Example output (path: ./PRSIMD_results_MRfactors):
+ CAD_LDpred2_auroc.txt: the AUROC on the test set
+ CAD_LDpred2_best_hyper.txt: the indexes of best alpha and beta 
+ CAD_LDpred2_y_pred.txt: the precited score on the test set
+ CAD_LDpred2_y_test.txt: the true disease label on the test set



## Contact  
For any question and suggestion, please feel free to contact with me or my supervisor Dr.Eric Zhang.  
(Yu XU, email: csyuxu@hkbu.edu.hk)  
(Lu Zhang, email: ericluzhang@hkbu.edu.hk) 
