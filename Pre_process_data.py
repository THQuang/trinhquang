import numpy as np
import os,sys
import time

char_list= ["H","C","N","O","F","P","S","Cl","Br","I",
"n","c","o","s",
"1","2","3","4","5","6","7","8",
"(",")","[","]",
"-","=","#","/","\\","+","@","X","Y"]


char_dict={'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'P': 5, 
'S': 6, 'Cl': 7, 'Br': 8, 'I': 9, 
'n': 10, 'c': 11, 'o': 12, 's': 13, 
'1': 14, '2': 15, '3': 16, '4': 17, '5': 18, '6': 19, '7': 20, '8': 21, 
'(': 22, ')': 23, '[': 24, ']': 25, '-': 26, '=': 27, '#': 28, 
'/': 29, '\\': 30, '+': 31, '@': 32, 'X': 33, 'Y': 34}



char_list1=list()
char_list2=list() 
char_dict1=dict()
char_dict2=dict()
for key in char_list:
    if len(key)==1:
        char_list1+=[key]
        char_dict1[key]=char_dict[key]
    elif len(key)==2:
        char_list2+=[key]
        char_dict2[key]=char_dict[key]
    else:
        print("strange ",key)

Nchar=len(char_list)
batch_size = 100
sample_size = 100
seq_length = 109
dev = 0.2
data_name = 'test'
data_dir ='/content/drive/My Drive/Genarate_smiles/ZINC'
smiles_filename=data_dir+"/" + data_name + ".txt"

fp = open(smiles_filename)
data_lines=fp.readlines()
fp.close()

smiles_list=[]
Maxsmiles=0
Xdata=[]
Ydata=[]
Ldata=[]
Pdata=[]
title=""
for line in data_lines:
    if line[0]=="#":
        title=line[1:-1]
        title_list=title.split() # title_list = [smi logP SAS QED MW TPSA]
        continue
    arr=line.split() #tách dữ liệu smiles trong từ dòng
    if len(arr)<2:
        continue
    smiles=arr[0] #smiles sẽ được add và list
    if len(smiles)>seq_length: # loại bỏ các chuỗi smiles có độ dài không mong muốn
        continue
    smiles0=smiles.ljust(seq_length,'Y') # Thêm ký tự 'Y' cho đủ độ dài chuỗi smile như mong muốn 110
    smiles_list+=[smiles]

    X_smiles='X'+smiles
    Y_smiles=smiles+'Y'
    X_d=np.zeros([seq_length+1],dtype=int)
    Y_d=np.zeros([seq_length+1],dtype=int)
    X_d[0]=char_dict['X']
    Y_d[-1]=char_dict['Y']

    Nsmiles=len(smiles)
    if Maxsmiles<Nsmiles:
        Maxsmiles=Nsmiles
    i=0
    istring=0
    check=True
    # ý tưởng: kiểm tra số lượng ký tự đơn và ký tự đôi trong chuỗi smiles
    while check:
        char2=smiles[i:i+2]
        char1=smiles[i]
        if char2 in char_list2 :
            j=char_dict2[char2]
            i+=2
            if i>=Nsmiles:
                check=False
        elif char1 in char_list1 :
            j=char_dict1[char1]
            i+=1
            if i>=Nsmiles:
                check=False
        else:
            print(char1,char2,"error")
            sys.exit()
        X_d[istring+1]=j #mảng chứa vị trí của các ký tự trong chuỗi smiles và X_d[0] = char_dict1('X) = 33
        Y_d[istring]=j # mảng chưa vị trí các ký tự trong chuỗi smiles
        istring+=1
    for i in range(istring,seq_length):
        X_d[i+1]=char_dict['Y'] #Thêm ký tự Y hoàn thành độ dài 110
        Y_d[i]=char_dict['Y']# 
    print(X_d)
    Xdata+=[X_d]
    Ydata+=[Y_d]
    Ldata+=[istring+1] # chiều dài thực của smiles

Xdata = np.asarray(Xdata,dtype="int32")
Ydata = np.asarray(Ydata,dtype="int32")
Ldata = np.asarray(Ldata,dtype="int32")
Pdata = np.asarray(Pdata,dtype="float32")
print(Xdata.shape,Ydata.shape,Ldata.shape)

data_dir2="/content/drive/My Drive/Genarate_smiles/ZINC/"
if not os.path.exists(data_dir2):
    os.makedirs(data_dir2)

Xfile=data_dir2+"X"+data_name+".npy"
Yfile=data_dir2+"Y"+data_name+".npy"
Lfile=data_dir2+"L"+data_name+".npy"
# Pfile=data_dir2+"P"+data_name+".npy"
np.save(Xfile,Xdata)
np.save(Yfile,Ydata)
np.save(Lfile,Ldata)
