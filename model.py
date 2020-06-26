import os,sys
import numpy as np
import torch
from torch.nn.parameter import Parameter

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Encoder(nn.Module):

    def __init__(self,para,bias=True):
        super(Encoder,self).__init__()

        self.Sequen_length=para['Nseq']
        self.Num_feature=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        #Layer Embedd
        self.embedd = nn.Embedding(self.Num_feature, self.Num_feature)

        # Layer LSTM
        self.encoder_rnn = nn.LSTM(input_size=self.Num_feature,hidden_size=self.hidden_dim,
                num_layers=self.NLSTM_layer,bias=True,
                batch_first=True,bidirectional=False)
        
        # Khơi tạo tham số 
        for param in self.encoder_rnn.parameters():
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

    def forward(self,X,L):

        batch_size=X.shape[0]
        device=X.device

        # Khởi tạo hidden_state và cell_state
        enc_hidden = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device) 
        enc_cell = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        X = self.embedd(X)

        out,(encoder_hn,encoder_cn)=self.encoder_rnn(X,(enc_hidden,enc_cell))
        
        ''' (L - 1): vị trí vector hidden_state khi chạy hết chiều dài thực của chuỗi trong khối lstm 
            * L mảng chứa giá trị chiều dài thực của các chuỗi smile: [33,  3, 27,  1, 22,  1,  1, 11, 14, 10, 10, 11, 22, 26, 11, 15, 11, 11,
                                                            11, 22,  7, 23, 11, 11, 15, 23, 12, 14, 23,  2, 14,  1, 24,  1, 32,  0,
                                                            25, 15,  1,  1, 27,  1,  1, 24,  1, 32,  0, 25, 15,  1, 14, 34, 34, 34,
                                                            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
                                                            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
                                                            34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34,
                                                            34, 34]
            * out  = batch chứa các hidden sau mỗi ký tự (110x300)
            
        '''
        # VỊ trí vector hidden cập nhật sau ký tự cuối cùng(tính trên chiều dài thật)
        last_step_index_list = (L - 1).view(-1, 1).expand(out.size(0), out.size(2)).unsqueeze(1) #100x1x300

        # Trả về vector hidden ở ví trí L0 - 1 trong vector out_of_encode = out(batch chứa các hidden_state được tính sau mỗi ký tự)
        Z = out.gather(1,last_step_index_list).squeeze()#100x300 

        Z= F.normalize(Z,p=2,dim=1)
        return Z

class Decoder(nn.Module):

    def __init__(self,para,bias=True):
        super(Decoder,self).__init__()

        self.Sequen_length=para['Nseq']
        self.Num_feature=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.embedd = nn.Embedding(self.Num_feature, self.Num_feature)# Embedding(35,35)

        # LSTM layer(335,300)
        self.decoder_rnn = nn.LSTM(input_size=self.Num_feature+self.hidden_dim,
            hidden_size=self.hidden_dim, num_layers=self.NLSTM_layer,
            bias=True, batch_first=True,bidirectional=False)
        
        # Khởi tạo tham số 
        for param in self.decoder_rnn.parameters(): # (1200,335), (1200,300), 1200, 1200
            if len(param.shape)>=2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

        self.linear=nn.Linear(self.hidden_dim,self.Num_feature) #Linear(300,35)
        # Khởi tạo tham số cho khổi linear(weight và bias)
        nn.init.xavier_normal_(self.linear.weight.data) # (35,300)
        nn.init.normal_(self.linear.bias.data) # 35

    def forward(self, Z, X0, L0):

        batch_size=Z.shape[0]
        device=Z.device
        dec_hidden = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        dec_cell = torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        X = self.embedd(X0) # bzx110x35
        
        # Vector Z bzx300 được repeat(dim 1) ==> vector bzx110x300
        Zm=Z.view(-1,1,self.hidden_dim).expand(-1,self.Sequen_length,self.hidden_dim) # Zm = bzx110x300 , Z = bzx300
        Z_pr= torch.cat((Zm,X),2) # bzx110x335 (concat giữa vector 1x35 và vector 1x300 ==> vector 1x335)
        dec_lstm_out,(decoder_hn,decoder_cn)=self.decoder_rnn(Z_pr,(dec_hidden,dec_cell))
        
        output_vector= self.linear(dec_lstm_out)
        return output_vector

    def genarate(self, Z):
        batch_size= Z.shape[0]
        device= Z.device
        dec_hidden= torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)
        dec_cell= torch.zeros(self.NLSTM_layer*1,batch_size,self.hidden_dim).to(device)

        # Khởi tạo ký tự step 0 = X(33)
        seq= torch.zeros([batch_size,1],dtype=torch.long).to(device)
        seq[:,0]= 33
        Y = seq # Vector 100x1 chứa giá trị index X(33)

        Zm=Z.view(-1,1,self.hidden_dim).expand(-1,1,self.hidden_dim) #Zm = repeat vector Z(bzx300) ==> Zm(bzx110x300)

        decoder_hn=dec_hidden
        decoder_cn=dec_cell
        for i in range(self.Sequen_length):
            dec_hidden=decoder_hn
            dec_cell=decoder_cn

            X = self.embedd(Y) # bzx1x35
            ZX=torch.cat((Zm,X),2) # bzx1x335
            dec_out,(decoder_hn,decoder_cn)=self.decoder_rnn(ZX,(dec_hidden,dec_cell)) 
            dec=self.linear(dec_out)
            Y= torch.argmax(dec,dim=2)

            seq=torch.cat((seq,Y),dim=1)
        return seq


class Generator(nn.Module):
    def __init__(self,para,bias=True):
        super(Generator,self).__init__()

        self.seed_dim=para['seed_dim']
        self.hidden_dim=para['hidden_dim']

        self.fc1=nn.Linear(self.seed_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)

        self.fc2=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)

        self.fc3=nn.Linear(self.hidden_dim,self.hidden_dim)
        nn.init.xavier_normal_(self.fc3.weight.data)
        nn.init.normal_(self.fc3.bias.data)

    def forward(self,S0):

        S1=self.fc1(S0)
        S1=torch.relu(S1)
        S2=self.fc2(S1)
        S2=torch.relu(S2)
        Zgen=self.fc3(S2)
        Zgen=F.normalize(Zgen,p=2,dim=1)
        return Zgen

class Discrimination(nn.Module):
    def __init__(self,para,bias=True):
        super(Discrimination,self).__init__()

        self.hidden_dim=para['hidden_dim']

        self.fc1=nn.Linear(self.hidden_dim,self.hidden_dim)# linear(300,300)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.normal_(self.fc1.bias.data)

        self.fc2=nn.Linear(self.hidden_dim,self.hidden_dim)# linear(300,300)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.normal_(self.fc2.bias.data)

        self.fc3=nn.Linear(self.hidden_dim,1)# linear(300,1)
        nn.init.xavier_normal_(self.fc3.weight.data)
        nn.init.normal_(self.fc3.bias.data)

    def forward(self,Z0):
        D1=self.fc1(Z0)
        D1=torch.relu(D1)
        D2=self.fc2(D1)
        D2=torch.relu(D2)
        Dout=self.fc3(D2)

        return Dout

    def clip(self,epsi=0.01):

        # Set up giá trị khởi tạo trong khoảng(-0.01,0.01)
        torch.clamp_(self.fc1.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.fc1.bias.data,min=-epsi,max=epsi)
        torch.clamp_(self.fc2.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.fc2.bias.data,min=-epsi,max=epsi)
        torch.clamp_(self.fc3.weight.data,min=-epsi,max=epsi)
        torch.clamp_(self.fc3.bias.data,min=-epsi,max=epsi)


class ARAE(nn.Module):

    def __init__(self,para,bias=True):
        super(ARAE,self).__init__()

        self.Sequen_length=para['Nseq']
        self.Num_feature=para['Nfea']

        self.hidden_dim=para['hidden_dim']
        self.NLSTM_layer=para['NLSTM_layer']

        self.Enc=Encoder(para)
        self.Dec=Decoder(para)
        self.Gen=Generator(para)
        self.Dis=Discrimination(para)


    def AE(self, X, L, noise):

        Z = self.Enc(X, L)
        Zn = Z+ noise #Z + Nhiễu gauss được khởi tạo
        decoded = self.Dec(Zn, X, L)
        return decoded
