#!/usr/bin/env python
import os,sys
import numpy as np

import torch
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

def accu(pred,val,batch_l):

    correct=0
    total=0
    cor_seq=0
    for i in range(0,batch_l.shape[0]):
        mm=(pred[i,0:batch_l[i]].cpu().data.numpy() == val[i,0:batch_l[i]].cpu().data.numpy())
        correct+=mm.sum()
        total+=batch_l[i].sum()
        cor_seq+=mm.all()
    acc=correct/float(total)
    acc2=cor_seq/batch_l.shape[0]
    return acc,acc2

def vec_to_char(out_num):
    stri=""
    for cha in out_num:
        stri+=char_list[cha]
    return stri

def cal_prec_rec(Ypred,Ydata,conf):

    small=0.0000000001
    Ypred0=Ypred.cpu().data.numpy()
    Ydata0=Ydata.cpu().data.numpy()
    Ypred00=Ypred0>conf
    mm=Ypred00*Ydata0
    TP=mm.sum()
    A=Ydata0.sum()
    P=Ypred00.sum()
    precision=(TP+small)/(P+small)
    recall=(TP+small)/A

    return precision, recall

class Encoder(nn.Module):

    def __init__(self,para,bias=True):
        super(Encoder,self).__init__()

        self.Nseq=para['Nseq']
        self.Nfea=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)
        self.encoder_rnn = nn.LSTM(input_size=self.Nfea,hidden_size=self.hidden_dim,
                num_layers=self.NLSTM_layer,bias=True,
                batch_first=True,bidirectional=False)

        for param in self.encoder_rnn.parameters():
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self,X0,L0):

        batch_size=X0.shape[0]
        device=X0.device
        enc_h0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        enc_c0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        X = self.embedd(X0)
        out,(encoder_hn,encoder_cn)=self.encoder_rnn(X,(enc_h0,enc_c0))
        last_step_index_list = (L0 - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1)
        Z=out.gather(1,last_step_index_list).squeeze()
#        Z=torch.sigmoid(Z)
        Z=F.normalize(Z,p=2,dim=1)

        return Z

class Decoder(nn.Module):

    def __init__(self,para,bias=True):
        super(Decoder,self).__init__()

        self.Nseq=para['Nseq']
        self.Nfea=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Nfea, self.Nfea)

#        self.decoder_rnn = nn.LSTM(input_size=self.Nfea,
        self.decoder_rnn = nn.LSTM(input_size=self.Nfea+self.hidden_dim,
            hidden_size=self.hidden_dim, num_layers=self.NLSTM_layer,
            bias=True, batch_first=True,bidirectional=False)

        for param in self.decoder_rnn.parameters():
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        self.decoder_fc1=nn.Linear(self.hidden_dim,self.Nfea)
        nn.init.xavier_normal_(self.decoder_fc1.weight.data)
        nn.init.normal_(self.decoder_fc1.bias.data)

    def forward(self, Z, X0, L0):

        batch_size=Z.shape[0]
        device=Z.device
        dec_h0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        X = self.embedd(X0)
        Zm=Z.view(-1,1,self.hidden_dim).expand(-1,self.Nseq,self.hidden_dim)
        ZX=torch.cat((Zm,X),2)

#        dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(X0,(Z.view(1,-1,self.hidden_dim),dec_c0))
        dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(ZX,(dec_h0,dec_c0))
        dec=self.decoder_fc1(dec_out)
        return dec

    def decoding(self, Z):
        batch_size=Z.shape[0]
        device=Z.device
        dec_h0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        dec_c0 = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        seq=torch.zeros([batch_size,1],dtype=torch.long).to(device)
        seq[:,0]=self.Nfea-2

#        Xdata_onehot=torch.zeros([batch_size,1,self.Nfea],dtype=torch.float32).to(device)
#        Xdata_onehot[:,0,self.Nfea-2]=1
        Y = seq
        Zm=Z.view(-1,1,self.hidden_dim).expand(-1,1,self.hidden_dim)

        decoder_hn=dec_h0
        decoder_cn=dec_c0
#        seq2=Xdata_onehot
        for i in range(self.Nseq):
            dec_h0=decoder_hn
            dec_c0=decoder_cn

            X = self.embedd(Y)
            ZX=torch.cat((Zm,X),2)
            dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(ZX,(dec_h0,dec_c0))
            dec=self.decoder_fc1(dec_out)
            Y= torch.argmax(dec,dim=2)
#            Xdata_onehot=torch.zeros([batch_size,self.Nfea],dtype=torch.float32).to(device)
#            Xdata_onehot=Xdata_onehot.scatter_(1,Y,1).view(-1,1,self.Nfea)
            seq=torch.cat((seq,Y),dim=1)
#            seq2=torch.cat((seq2,dec),dim=1)

        return seq #, seq2[:,1:]

class Generator(nn.Module):
    def __init__(self,para,bias=True):
        super(Generator,self).__init__()

        self.seed_dim=para['seed_dim']
        self.hidden_dim=para['hidden_dim']

        self.generator_fc1=nn.Linear(self.seed_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.generator_fc1.weight.data)
        nn.init.normal_(self.generator_fc1.bias.data)

        self.generator_fc2=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.generator_fc2.weight.data)
        nn.init.normal_(self.generator_fc2.bias.data)

        self.generator_fc3=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.generator_fc3.weight.data)
        nn.init.normal_(self.generator_fc3.bias.data)

    def forward(self,S0):

        S1=self.generator_fc1(S0)
        S1=torch.relu(S1)
        S2=self.generator_fc2(S1)
        S2=torch.relu(S2)
        Zgen=self.generator_fc3(S2)
#        Zgen=torch.sigmoid(Zgen)
        Zgen=F.normalize(Zgen,p=2,dim=1)

        return Zgen

class Critic(nn.Module):
    def __init__(self,para,bias=True):
        super(Critic,self).__init__()

        self.hidden_dim=para['hidden_dim']

        self.critic_fc1=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.critic_fc1.weight.data)
        nn.init.normal_(self.critic_fc1.bias.data)

        self.critic_fc2=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.critic_fc2.weight.data)
        nn.init.normal_(self.critic_fc2.bias.data)

        self.critic_fc3=nn.Linear(self.hidden_dim,1)
        nn.init.xavier_normal_(self.critic_fc3.weight.data)
        nn.init.normal_(self.critic_fc3.bias.data)

    def forward(self,Z0):

        D1=self.critic_fc1(Z0)
        D1=torch.relu(D1)
        D2=self.critic_fc2(D1)
        D2=torch.relu(D2)
        Dout=self.critic_fc3(D2)

        return Dout

    def clip(self,epsi=0.01):
        torch.clamp_(self.critic_fc1.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc1.bias.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc2.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc2.bias.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc3.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.critic_fc3.bias.data,min=-epsi,max=epsi)


class ARAE(nn.Module):

    def __init__(self,para,bias=True):
        super(ARAE,self).__init__()

        self.Nseq=para['Nseq']
        self.Nfea=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.Enc=Encoder(para)
        self.Dec=Decoder(para)
        self.Gen=Generator(para)
        self.Cri=Critic(para)


    def AE(self, X0, L0, noise):

        Z = self.Enc(X0, L0)
#        print(Z.shape, noise.shape)
        Zn = Z+noise
        decoded = self.Dec(Zn, X0, L0)

        return decoded




import os,sys
import numpy as np
import math

import torch
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

char_list= ["H","C","N","O","F","P","S","Cl","Br","I",
"n","c","o","s",
"1","2","3","4","5","6","7","8",
"(",")","[","]",
"-","=","#","/","\\","+","@","<",">"]


char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5,
'S': 6, 'Cl': 7, 'Br': 8, 'I': 9,
'n': 10, 'c': 11, 'o': 12, 's': 13,
'1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '8': 21,
'(': 22, ')': 23, '[': 24, ']': 25, '-': 26, '=': 27, '#': 28,
'/': 29, '\\': 30, '+': 31, '@': 32, '<': 33, '>': 34}
def vec_to_char(out_num):
    stri=""
    for cha in out_num:
        stri+=char_list[cha]
    return stri



def main():


    Ntest=10000

    Nfea=len(char_list)

    Nseq=110
    hidden_dim=300
    seed_dim=hidden_dim
    NLSTM_layer=1
    batch_size=100

    N_batch=int(math.ceil(Ntest/batch_size))
    print(N_batch,Ntest)

    use_cuda= torch.cuda.is_available()
    if use_cuda==True:
        device_num=torch.cuda.current_device()
        print(device_num)
        device = torch.device("cuda:%d" %device_num)
        torch.set_num_threads(2)
    else :
        device =  torch.device("cpu")
        torch.set_num_threads(24)
#    device =  torch.device("cpu")
    print(device)


    para={'Nseq':Nseq, 'Nfea':Nfea, 'hidden_dim':hidden_dim,
            'seed_dim':seed_dim,'NLSTM_layer':NLSTM_layer,'device':device}

    model=ARAE(para)
    model.to(device)

    save_dir="/content/drive/My Drive/ARAE_new/save_ARAE"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_result_dir="/content/drive/My Drive/ARAE_new/result_ARAE_gen"
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)

    total_st = time.time()

    std0=0.0
    std=0.0
    std_seed=0.25
    std_decay_ratio=0.99
    mean0=torch.zeros(batch_size,hidden_dim)
    mean_seed=torch.zeros(batch_size,seed_dim)

#    epoch_list=[9]
    epoch_list=[40]

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

            # batch_s=torch.normal(mean=mean_seed,std=std_seed).to(device)
            Z_gen=torch.randn((100,300)).to(device)
            print(Z_gen.shape)

            out_num_ARAE=model.Dec.decoding(Z_gen)

#            for k in range(0,10):
#                out_string=vec_to_char(out_num_ARAE[k])
#                print("ARAE: ",out_string)

            for k in range(0,b_size):
                line_ARAE=vec_to_char(out_num_ARAE[k])+"\n"
                fp_ARAE.write(line_ARAE)
        fp_ARAE.close()

        et=time.time()
        print("time: %10.2f" %(et-st))


    print('Finished Training')
    total_et = time.time()
    print ("time : %10.2f" %(total_et-total_st))





if __name__=="__main__":
    main()