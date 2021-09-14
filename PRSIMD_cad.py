#!/usr/bin/python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import time
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import os
import datatools as dt


# model definition
class model(nn.Module):
    # define model elements
    def __init__(self, n_features):
        super(model, self).__init__()

        # weight for numeric feature
        self.w_age = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))
        self.w_bmi = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))
        self.w_phys_mode = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))
        self.w_phys_vigo = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))  
        self.w_bp_Diastolic = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))  
        self.w_bp_Systolic = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))  
        self.w_blood_CystatinC = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))
        self.w_blood_Urea = nn.Parameter(torch.FloatTensor(torch.randn(2,1))) 
        self.w_blood_Cholesterol = nn.Parameter(torch.FloatTensor(torch.randn(2,1))) 
        self.w_blood_Triglycerides = nn.Parameter(torch.FloatTensor(torch.randn(2,1))) 
        self.w_blood_HDLcholesterol = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))
        # yes: 1 or no: 0 . If not, w*0
        self.w_family_heart = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))  
        self.w_under_diabetes = nn.Parameter(torch.FloatTensor(torch.randn(2,1)))             
        # weight for categorical feature
        self.w_gender = nn.Parameter(torch.FloatTensor(torch.randn(2,2)))                       
        self.w_smoke = nn.Parameter(torch.FloatTensor(torch.randn(2,3))) # 
        self.w_edu = nn.Parameter(torch.FloatTensor(torch.randn(2,7))) # UKBB Data-Coding 100305
        # confidence matrix for every features
        self.M = nn.Parameter(torch.FloatTensor(torch.randn(2,n_features)))
        
        self.sigmoid = torch.sigmoid
        
    # forward propagation input
    def forward(self,f_data):
        # Age when recruitment
        age = f_data[:,2].unsqueeze(0)/10 # age group every 10 as interval
        age = age.int().float()
        age = self.w_age.reshape(2,1) @ age  
        age = torch.sigmoid(age)
        age = self.M[:,0].reshape(2,1) * age # v_gamma * const()
        # BMI
        bmi = f_data[:,1].unsqueeze(0)
        bmi = self.w_bmi.reshape(2,1) @ bmi
        bmi = torch.sigmoid(bmi)
        bmi = self.M[:,1].reshape(2,1) * bmi
        # Gender
        gender = dt.one_hot(f_data[:,0].unsqueeze(0),2)
        gender = self.w_gender @ gender.T # v_w
        gender = torch.sigmoid(gender)
        gender = self.M[:,2].reshape(2,1) * gender # v_gamma * const()
        # Smoke status
        smoke = dt.one_hot(f_data[:,3].unsqueeze(0),3)
        smoke = self.w_smoke @ smoke.T
        smoke = torch.sigmoid(smoke)
        smoke = self.M[:,3].reshape(2,1) * smoke
        # Physical activity
        phys_mode = f_data[:,4].unsqueeze(0)
        phys_vigo = f_data[:,5].unsqueeze(0)
        physical = self.w_phys_mode.reshape(2,1) @ phys_mode + self.w_phys_vigo.reshape(2,1) @ phys_vigo
        physical = torch.sigmoid(physical)
        physical = self.M[:,4].reshape(2,1) * physical
        # Blood pressure
        Diastolic = f_data[:,6].unsqueeze(0)
        Systolic = f_data[:,7].unsqueeze(0)
        bpress = self.w_bp_Diastolic.reshape(2,1) @ Diastolic + self.w_bp_Systolic.reshape(2,1) @ Systolic
        bpress = torch.sigmoid(bpress)
        bpress = self.M[:,5].reshape(2,1) * bpress
        # Blood chemistry
        blood_CystatinC = f_data[:,11].unsqueeze(0)
        blood_Cholesterol = f_data[:,8].unsqueeze(0)
        blood_Urea = f_data[:,12].unsqueeze(0)
        blood_Triglycerides = f_data[:,9].unsqueeze(0)
        blood_HDLcholesterol = f_data[:,10].unsqueeze(0)
        blood_assays = (self.w_blood_CystatinC.reshape(2,1) @ blood_CystatinC +
                        self.w_blood_Cholesterol.reshape(2,1) @ blood_Cholesterol+
                        self.w_blood_Urea.reshape(2,1) @ blood_Urea+
                        self.w_blood_Triglycerides.reshape(2,1) @ blood_Triglycerides+
                        self.w_blood_HDLcholesterol.reshape(2,1) @ blood_HDLcholesterol)
        blood_assays = torch.sigmoid(blood_assays)
        blood_assays = self.M[:,6].reshape(2,1) * blood_assays     
        # Family history
        family_heart = f_data[:,13].unsqueeze(0)
        family_history = self.w_family_heart.reshape(2,1) @ family_heart
        family_history = torch.sigmoid(family_history)
        family_history = self.M[:,7].reshape(2,1) * family_history
        # Underlying diseases
        Ud_Diabetes = f_data[:,14].unsqueeze(0)
        underly_dise = self.w_under_diabetes.reshape(2,1) @ Ud_Diabetes 
        underly_dise = torch.sigmoid(underly_dise)
        underly_dise = self.M[:,8].reshape(2,1) * underly_dise
        # Education
        education = dt.one_hot(f_data[:,15].unsqueeze(0),7)
        education = self.w_edu @ education.T 
        education = torch.sigmoid(education)
        education = self.M[:,9].reshape(2,1) * education

        f=(
            age + gender + bmi + smoke + physical + bpress + blood_assays + family_history + underly_dise + education
        )
   
        f = torch.exp(f)
        f_sum = torch.sum(f,0)
        Prob = f/f_sum
        return Prob


# Logistic model definition
class PRS_model(nn.Module):
    def __init__(self):
        super(PRS_model, self).__init__()
        
        self.w = nn.Parameter(torch.FloatTensor(torch.randn(1))) 
        self.b = nn.Parameter(torch.FloatTensor(torch.randn(1))) 
        
    def forward(self,score):
        logit = 1/(1+torch.exp(-(self.w * score + self.b)))     
        return logit

# model training
def model_train(data,label,PRSore,val_data,val_label,val_PRSore,num_features,
                param_save,train_loss_save,val_loss_save,alpha=1,beta=1,batch_size=100,n_epoch=1000):    
   
    # model instances
    pred = model(num_features)
    optimizer = torch.optim.Adadelta(pred.parameters(), rho = 0.95, weight_decay = 0.001)
    prs = PRS_model()
    optimizer_prs = torch.optim.Adadelta(prs.parameters(), rho = 0.95, weight_decay = 0.001)

    n_batches = int(np.ceil(float(len(label)) / float(batch_size)))
    
    val_loss_best = 100000
    val_acc_best = 0.0
    val_loss_epoch = []
    train_loss_epoch = []

    start_time = time.time()    
    
    # training k times epoch
    for epoch in range(n_epoch):
        loss_batch = []
        batches = random.sample(range(n_batches), n_batches) # suf batch index 
        for i in batches:
            # read batch data
            batch_y = label[batch_size * i : batch_size * (i + 1)]
            batch_data = data[batch_size * i : batch_size * (i + 1)]
            batch_PRS= PRSore[batch_size * i : batch_size * (i + 1)]
       
            optimizer.zero_grad() 
            optimizer_prs.zero_grad() 

            result_prs = prs(batch_PRS)
            result = pred(batch_data).T  # reshape [batch_size, n_class]
            # loss calculation
            kl = dt.kl_loss_score(result[:,1], result_prs) # KL-Divergence , PRS model is fixed 
            loss_f=F.cross_entropy(result, batch_y) 
            loss_prs = -torch.mean(batch_y*torch.log(result_prs)+(1-batch_y)*torch.log(1-result_prs))
            loss =loss_prs + alpha * kl + beta * loss_f
            loss.backward()
            # params update
            optimizer.step()
            optimizer_prs.step()
            # loss information every batch
            loss_batch.append(loss.data) 
        # training loss information every epoch
        train_loss_epoch.append(np.mean(loss_batch)) 

        if np.mod(epoch, 50) == 0:
            count_time = time.time()-start_time
            print("run epoch:%d , time consumed: %d s" % (epoch,int(count_time) ))
        
        val_acc = validation_acc(prs, pred, val_data, val_label, val_PRSore, batch_size, alpha, beta)
        if val_acc > val_acc_best:
            val_acc_best = val_acc
            state = {'pred':pred.state_dict(), 'optimizer':optimizer.state_dict(),
            'prs':prs.state_dict(), 'optimizer_prs':optimizer_prs.state_dict(), 'epoch':epoch,'alpha':alpha,'beta':beta}
            torch.save(obj= state, f= param_save+'model_param_g_cad_'+str(alpha)+'_'+str(beta)+'.pth')
    # for epoch error plot     
    np.save(train_loss_save+'train_loss_cad_'+str(alpha)+'_'+str(beta), train_loss_epoch)
    np.save(val_loss_save+'val_loss_cad_'+str(alpha)+'_'+str(beta), val_loss_epoch)


def validation(prs_model,f_model,data_val,label_val,PRSore_val,batch_size,alpha,beta):
    n_batches = int(np.ceil(float(len(label_val)) / float(batch_size)))
    loss_batch_val = []
    for i in range(n_batches):
        batch_y = label_val[batch_size * i : batch_size * (i + 1)]
        batch_data = data_val[batch_size * i : batch_size * (i + 1)]
        batch_PRS= PRSore_val[batch_size * i : batch_size * (i + 1)]
        # input
        val_result = f_model(batch_data).T
        val_result_prs = prs_model(batch_PRS)
        # loss calc
        kl = dt.kl_loss_score(val_result[:,1], val_result_prs)  
        loss_f=F.cross_entropy(val_result, batch_y) 
        loss_prs = -torch.mean(batch_y*torch.log(val_result_prs)+(1-batch_y)*torch.log(1-val_result_prs))
        loss =loss_prs + alpha * kl + beta * loss_f
        loss_batch_val.append(loss.data)

    loss_mean = np.mean(loss_batch_val)

    return loss_mean
    
def validation_acc(prs_model,f_model,data_val,label_val,PRSore_val,batch_size,alpha,beta):
  
    val_result = f_model(data_val).T
    val_result_prs = prs_model(PRSore_val)
    PRSPR_Pred = val_result[:,1] + val_result_prs
    pred_result = PRSPR_Pred.detach().numpy()
    threshold =np.mean(pred_result)  # simply set  a threshold
    classification = []
    for i in range(len(label_val)):
        if pred_result[i] >= threshold:
            classification.append(1)
        else:
            classification.append(0)

    val_acc = torch.sum(torch.tensor(classification) == label_val)
    final_val_acc = val_acc/len(label_val)

    return final_val_acc
    
def evaluation_auc(PRS_model,model,data,label,PRS,param_path,alpha_test,beta_test,sets='noSpeci',saveResult=False,result_folder=None):

    trained_param = param_path+'model_param_g_cad_'+str(alpha_test)+'_'+str(beta_test)+'.pth'
    reload_states = torch.load(trained_param)

    Features_model = model
    PRS_logistic = PRS_model
    PRS_logistic.load_state_dict(reload_states['prs'])
    Features_model.load_state_dict(reload_states['pred'])
    PRSPR_Pred= PRS_logistic(PRS) + Features_model(data).T[:,1]
    result_list = PRSPR_Pred.detach().numpy()
    label_list = label.detach().numpy()
    if saveResult == 'yes':       
        np.save(result_folder+'/y_test',label)
        np.save(result_folder+'/y_pred',result_list)
        f1 = open(result_folder+'/y_test.txt','w')
        f2 = open(result_folder+'/y_pred.txt','w')           
        for i in range(len(label_list)):
            f1.write(str(label_list[i])+'\n')
            f2.write(str(result_list[i])+'\n')
        f1.close()
        f2.close()
    auc =roc_auc_score(label, result_list)
   
    return auc

if __name__ == '__main__':
    torch.set_num_threads(10)
    work_path = os.getcwd()
    ## RPS score Loading ##
    PRS_tr = work_path + '/PRS/gw_train_cad.score'
    PRS_val = work_path + '/PRS/gw_vali_cad.score'
    PRS_te = work_path + '/PRS/gw_test_cad.score'
    sc_train, sc_train_size = dt.readbyLines(PRS_tr,datatype="float")
    sc_val, sc_val_size = dt.readbyLines(PRS_val,datatype="float")
    sc_test, sc_test_size = dt.readbyLines(PRS_te,datatype="float")
    # Pheno features loading ##
    train = np.load(work_path + '/dataset/train_cad.npy')
    val = np.load(work_path + '/dataset/validation_cad.npy')
    test = np.load(work_path + '/dataset/test_cad.npy')
    # labels  loading ##
    y_train = np.load(work_path + '/dataset/train_cad_y.npy')
    y_val = np.load(work_path + '/dataset/validation_cad_y.npy')
    y_test = np.load(work_path + '/dataset/test_cad_y.npy')
    # Torch data format # 
    train_data = Variable(torch.FloatTensor(train))
    val_data = Variable(torch.FloatTensor(val))
    test_data = Variable(torch.FloatTensor(test))
    PRS_test = Variable(torch.FloatTensor(sc_test))
    PRS_train = Variable(torch.FloatTensor(sc_train))
    PRS_val = Variable(torch.FloatTensor(sc_val))
    y_tr = Variable(torch.LongTensor(y_train.astype('long'))) 
    y_va = Variable(torch.LongTensor(y_val.astype('long')))
    y_te = Variable(torch.LongTensor(y_test.astype('long')))
    print('Data loading succeed')

    mode = input("train or evaluation mode?\t")

    inputFeatures= 10 # feature categories.
    
    # result folder 
    result = work_path + '/result'
    if result is not None:
        if not os.path.isdir(result):
            os.mkdir(result)
    result_folder = work_path + '/result/cad_200epoch_monitor_acc'
    if result_folder is not None:
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
    param_path = result_folder+ '/param/'
    if param_path is not None:
        if not os.path.isdir(param_path):
            os.mkdir(param_path)
    trainloss_path = result_folder+ '/loss/'
    valloss_path =  result_folder+ '/loss/'
    if trainloss_path is not None:
        if not os.path.isdir(trainloss_path):
            os.mkdir(trainloss_path)
    
    alpha_candi=[1,0.1,0.05,0.01,0.005]
    beta_candi=[1,0.1,0.05,0.01,0.005]

    error_input = False

    if mode=='train': 
        auc_txt = open(result_folder+'/auc_collected.txt','w')
        auc_txt.write('alpha\tbeta\tAUROC\n')
        for i in range(len(alpha_candi)):
            for j in range(len(beta_candi)):
                # train model
                model_train(data=train_data,label=y_tr,PRSore=PRS_train,val_data=val_data,val_label=y_va,val_PRSore=PRS_val,
                    num_features=inputFeatures,param_save=param_path,train_loss_save=trainloss_path,val_loss_save=valloss_path,
                    alpha=alpha_candi[i],beta=beta_candi[j],n_epoch=200)             
                # test after trainning
                eval_Fmodel = model(inputFeatures)
                eval_PRSmodel = PRS_model()
                auc_test = evaluation_auc(eval_PRSmodel,eval_Fmodel,test_data,y_te,PRS_test,param_path,alpha_candi[i],beta_candi[j],sets='test',saveResult='no')
                print("test auc:",auc_test)
                auc_txt.write(str(alpha_candi[i])+'\t'+str(beta_candi[j])+'\t'+ str(auc_test)+'\n')                                           
        auc_txt.close()  
        
    elif mode=='evaluation':
        alpha_input = input('alpha select(1,0.1,0.05,0.01,0.005)\t')
        if float(alpha_input) not in alpha_candi:
            print('no this alpha value')
            error_input=True
            
        beta_input = input('beta select(1,0.1,0.05,0.01,0.005)\t')
        if float(beta_input) not in beta_candi:
            print('no this beta value')
            error_input=True

        saveEva = input('save result?(yes/no)\t')
        if error_input == False:    
            eval_Fmodel = model(inputFeatures)
            eval_PRSmodel = PRS_model()
            auc_test = evaluation_auc(eval_PRSmodel,eval_Fmodel,test_data,y_te,PRS_test,param_path,alpha_input,beta_input,sets='test',saveResult=saveEva,result_folder=result_folder)
   
            print("test auc:",auc_test)
        else:
            print('alpha/beta input error')
    else:
        print("input error,please restart")
