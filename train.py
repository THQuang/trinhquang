
import os,sys
import numpy as np
import math
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ARAE as ARAE
from utils import *
import time

def main():

    char_list= ["H","C","N","O","F","P","S","Cl","Br","I",
    "n","c","o","s",
    "1","2","3","4","5","6","7","8",
    "(",")","[","]",
    "-","=","#","/","\\","+","@","X","Y"]

    # Đường dẫn chứa dữ liệu 
    datadir="/content/drive/My Drive/Genarate_smiles/ZINC/"

    # Đường dẫn lưu dữ liệu
    save_dir="/content/drive/My Drive/ARAE_new/save_ARAE"
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    Num_feature= len(char_list)
    train_data= Loaddataset(datadir,"train")
    test_data= Loaddataset(datadir,"test")

    Ntrain= len(train_data)
    Ntest= len(test_data)

    Sequen_length= 110
    hidden_dim=300
    seed_dim=hidden_dim
    NLSTM_layer=1
    batch_size=100

    conf=0.5 # Hệ số so sánh cal_prec_rec


    use_cuda= torch.cuda.is_available()
    if use_cuda==True:
        device_num=torch.cuda.current_device()
        device = torch.device("cuda:%d" %device_num)
    else :
        device =  torch.device("cpu")


    para={'Nseq':Sequen_length, 'Nfea':Num_feature, 'hidden_dim':hidden_dim,
            'seed_dim':seed_dim,'NLSTM_layer':NLSTM_layer,'device':device}

    model= ARAE(para) # Khai báo model 
    start_epoch=0

    #path=save_dir+"/save_%d.pth" %start_epoch # Đường dẫn chứa model đang upload 
    # model=torch.load(path)
    # model.load_state_dict(torch.load(path))
    

    model.to(device)
    # model.eval()

    total_st = time.time()

    criterion_AE = nn.CrossEntropyLoss()

    AE_parameters=list(model.Enc.parameters())+list(model.Dec.parameters())

    optimizer_AE = optim.Adam(AE_parameters, lr=0.001)
    optimizer_gen = optim.Adam(model.Gen.parameters(), lr=0.00001)
    optimizer_dis = optim.Adam(model.Dis.parameters(), lr=0.000002)


    N_batch=int(Ntrain/batch_size) # số batch dữ liệu 

    std0=0.2
    std00=0.02
    std_seed=0.25
    std_decay_ratio=0.99
    mean0=torch.zeros(batch_size,hidden_dim) # bzx300
    mean_seed=torch.zeros(batch_size,seed_dim) # bzx300

    Ngan=4
    Ndis=5
    for epoch in range(start_epoch,50):

        running_loss_AE=0.0
        running_loss_gen=0.0
        running_loss_dis=0.0

        st=time.time()

        std=std0*np.power(std_decay_ratio,epoch) + std00 #cập nhật hệ số std sau mỗi epoch

        train_loader=DataLoader(dataset=train_data,batch_size=batch_size,
            shuffle=True, drop_last=True, num_workers=2)
        Ntrain_batch=len(train_loader)
        model.train()
        for i, data in enumerate(train_loader):

            batch_x, batch_l = data
            batch_x     = batch_x.to(device)
            batch_l     = batch_l.to(device)

            batch_x2 = batch_x[:,1:] #target = x[1:-1] (chuỗi x sẽ bắt đầu từ ký tự X(33), chuỗi target là chuỗi không bắt đầu với X(33))

            optimizer_AE.zero_grad()
            noise = torch.normal(mean=mean0,std=std).to(device) # Nhiễu cộng vòa với Z
            out_decode = model.AE(batch_x, batch_l, noise) # out_decode(bzx110x35) = output của AE

            out = out_decode[:,:-1] # (Lược bỏ đi 1 giá trị padding ở cuối dãy ==> tạo ra vector có kích thước bằng với target = 109x35)
            loss_AE=criterion_AE(out.reshape(-1,Num_feature),batch_x2.reshape(-1))
            loss_AE.backward(retain_graph=True)
            optimizer_AE.step()
            running_loss_AE+=loss_AE.data


            Z_real=model.Enc(batch_x,batch_l)
            for i_gan in range(0,Ngan):
                for i_cri in range(0,Ndis):

                    batch_noise=torch.normal(mean=mean_seed,std=std_seed).to(device) # vector nhiễu bzx300
                    Z_fake=model.Gen(batch_noise)
                    D_fake=model.Dis(Z_fake)
                    D_real=model.Dis(Z_real)

                    optimizer_dis.zero_grad()
                    loss_dis= - D_real.mean() + D_fake.mean()
                    loss_dis.backward(retain_graph=True)
                    optimizer_dis.step()
                    running_loss_dis+=loss_dis.data/(Ndis*Ngan)
                    model.Dis.clip(0.01)

                batch_noise=torch.normal(mean=mean_seed,std=std_seed).to(device) # vector nhiễu bzx300
                Z_fake=model.Gen(batch_noise)
                D_fake=model.Dis(Z_fake)

                optimizer_gen.zero_grad()
                loss_gen= - D_fake.mean()
                loss_gen.backward(retain_graph=True)
                optimizer_gen.step()
                running_loss_gen+=loss_gen.data/Ngan 

            if i%100 == 0 :
                _,out_num_AE= torch.max(out_decode,2)
                acc, acc2= accu(out_num_AE,batch_x2,batch_l)
                print("reconstruction accuracy:", acc,acc2)

                out_num_ARAE=model.Dec.genarate(Z_fake)

                for k in range(0,2):
                    out_string=vec_to_char(batch_x2[k])
                    print("Smile_input         :",out_string)
                    out_string=vec_to_char(out_num_AE[k])
                    print("Smile_reconstruced  : ",out_string)
                for k in range(0,10):
                    out_string=vec_to_char(out_num_ARAE[k])
                    print("ARAE genarate smile : ",out_string)

        line_out="%d train loss: AE %6.3f dis %6.3f gen %6.3f" %(epoch,
                    running_loss_AE/Ntrain_batch,
                   running_loss_dis/Ntrain_batch,
                   running_loss_gen/Ntrain_batch)
        print(line_out)
        # Test model tren tap test
        loss_sum=[]
        loss_AE_test_sum=0
        loss_gen_test_sum=0
        loss_real_test_sum=0
        loss_dis_test_sum=0

        st=time.time()

        test_loader=DataLoader(dataset=test_data,batch_size=batch_size,
            shuffle=False, drop_last=False, num_workers=2)
        model.eval()
        for i, data in enumerate(test_loader):

            batch_x, batch_l = data
            batch_x     = batch_x.to(device)
            batch_l     = batch_l.to(device)

            batch_x2 = batch_x[:,1:]
            b_size=batch_x.shape[0]

            noise=torch.normal(mean=mean0,std=std).to(device)
            out_decoding = model.AE(batch_x,batch_l, noise)
            out2 = out_decoding[:,:-1]
            _,out_num_AE= torch.max(out_decoding,2)
            loss_AE_test=criterion_AE(out2.reshape(-1,Num_feature),batch_x2.reshape(-1)).data
            loss_AE_test_sum+=loss_AE_test*b_size


            Z_real = model.Enc(batch_x, batch_l)

            batch_s=torch.normal(mean=mean_seed,std=std_seed).to(device)
            Z_fake=model.Gen(batch_s)
            D_fake=model.Dis(Z_fake)
            D_real=model.Dis(Z_real)

            loss_gen_test = -D_fake.mean().data
            loss_real_test = D_real.mean().data
            loss_dis_test = (-D_real.mean() + D_fake.mean()).data

            loss_gen_test_sum += loss_gen_test*b_size
            loss_real_test_sum+= loss_real_test*b_size
            loss_dis_test_sum += loss_dis_test*b_size

            out_num_ARAE=model.Dec.genarate(Z_fake)


        loss_AE_test      = loss_AE_test_sum/Ntest
        loss_gen_test = loss_gen_test_sum/Ntest
        loss_real_test = loss_real_test_sum/Ntest
        loss_dis_test = loss_dis_test_sum/Ntest

        acc, acc2= accu(out_num_AE,batch_x2,batch_l)

        line_out="%d test: AE %6.3f gen %6.3f cri %6.3f real %6.3f " %(epoch,
                loss_AE_test, loss_gen_test, loss_cri_test, loss_real_test)
        print(line_out)

        et=time.time()
        print("time: %10.2f" %(et-st))
        
        # path=save_dir+"/save_%d.pth" %(epoch+17)
        # torch.save(model,path)


    print('Finished Training')
    total_et = time.time()
    print ("time : %10.2f" %(total_et-total_st))





if __name__=="__main__":
    main()
