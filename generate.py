#!/usr/bin/env python
import os,sys
import numpy as np
import math
from model import ARAE as ARAE
from utils import char_list, char_dict, Device, vec_to_char
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# đường dẫn chứa model đã trainning
save_dir="/content/drive/My Drive/ARAE_new/save_ARAE"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# Đường dẫn chứa chứa dữ liệu được generate của mỗi pre_train model
save_result_dir="/content/drive/My Drive/ARAE_new/result_ARAE_gen"
if not os.path.exists(save_result_dir):
    os.makedirs(save_result_dir)
    
def main():
    Ntest=100000 # Số lượng chất cần generate 
    Nfea=len(char_list)
    Nseq=110
    hidden_dim=300
    seed_dim=hidden_dim
    NLSTM_layer=1
    batch_size=100

    N_batch=int(math.ceil(Ntest/batch_size))
    print(N_batch,Ntest)

    use_cuda= torch.cuda.is_available()
    device= Device(use_cuda)
    
    para={'Nseq':Nseq, 'Nfea':Nfea, 'hidden_dim':hidden_dim,
            'seed_dim':seed_dim,'NLSTM_layer':NLSTM_layer,'device':device}

    model=ARAE(para)
    model.to(device)
    
    
    std_seed=0.25
    mean_seed=torch.zeros(batch_size,seed_dim)
    
    total_st = time.time()
    epoch_list=[1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,35,40,45,50]

    for epoch in epoch_list:
        path=save_dir+"/save_%d.pth" %epoch
        model=torch.load(path)
        model.to(device)

        st=time.time()
        save_result_dir2=save_result_dir+"/epoch%d" %(epoch + 1)
        if not os.path.exists(save_result_dir2):
            os.makedirs(save_result_dir2)

        file_ARAE=save_result_dir2+"/smiles_fake.csv"
        fp_ARAE=open(file_ARAE,"w")

        model.eval()
        for i in range(0,N_batch):
            ini=batch_size*i
            fin=batch_size*(i+1)
            if fin>Ntest:
                fin=Ntest
            mm=np.arange(ini,fin)
            b_size=mm.shape[0]
                
            batch_s=torch.normal(mean=mean_seed,std=std_seed).to(device)
            Z_gen=model.Gen(batch_s)
            
            out_num_ARAE=model.Dec.decoding(Z_gen) # Gennerate vector smile 

            for k in range(0,b_size):
                line_ARAE=vec_to_char(out_num_ARAE[k],char_list)+"\n"
                fp_ARAE.write(line_ARAE)
        fp_ARAE.close()
        et=time.time()
        print("time: %10.2f" %(et-st))

    print('Finished Generate')
    total_et = time.time()
    print ("time : %10.2f" %(total_et-total_st))

if __name__=="__main__":
    main()
