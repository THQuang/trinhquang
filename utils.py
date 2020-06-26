
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


class Loaddataset(Dataset):
    def __init__(self,datadir,dname):

        Xdata_file=datadir+"/X"+dname+".npy"
        self.Xdata=torch.tensor(np.load(Xdata_file),dtype=torch.long)
        Ldata_file=datadir+"/L"+dname+".npy"
        self.Ldata=torch.tensor(np.load(Ldata_file),dtype=torch.long)


        self.len = self.Xdata.shape[0]

    def __getitem__(self,index):
        return (self.Xdata[index], self.Ldata[index])


    def __len__(self):
        return self.len

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

def vec_to_char(out_num,char_list):
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
