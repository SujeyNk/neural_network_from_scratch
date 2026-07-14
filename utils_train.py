import torch, time
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

class AverageMeter:
    def __init__(self): self.sum=0; self.n=0
    def update(self,val,count=1): self.sum+=val*count; self.n+=count
    @property
    def avg(self): return self.sum/max(self.n,1)


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train(); loss_m=AverageMeter(); correct=0; total=0
    for x,y in tqdm(loader,leave=False):
        x,y=x.to(device),y.to(device)
        optimizer.zero_grad()
        logits=model(x)
        loss=loss_fn(logits,y)
        loss.backward(); optimizer.step()
        loss_m.update(loss.item(), x.size(0))
        if logits.ndim>1 and logits.size(1)>1:
            pred=logits.argmax(1)
        else:
            pred=(logits.squeeze()>0).long()
        correct+= (pred==y).sum().item(); total+=y.numel()
    return loss_m.avg, correct/total
 

def eval_epoch(model, loader, loss_fn, device):
    model.eval(); loss_m=AverageMeter(); correct=0; total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            logits=model(x)
            loss=loss_fn(logits,y)
            loss_m.update(loss.item(), x.size(0))
            if logits.ndim>1 and logits.size(1)>1:
                pred=logits.argmax(1)
            else:
                pred=(logits.squeeze()>0).long()
            correct+= (pred==y).sum().item(); total+=y.numel()
    return loss_m.avg, correct/total


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):
    model.to(device)
    hist=[]
    for ep in range(1,epochs+1):
        t0=time.time()
        tr_loss,tr_acc=train_epoch(model,train_loader,loss_fn,optimizer,device)
        va_loss,va_acc=eval_epoch(model,val_loader,loss_fn,device)
        t1=time.time()
        hist.append({'epoch':ep,'train_loss':tr_loss,'val_loss':va_loss,'train_acc':tr_acc,'val_acc':va_acc,'sec':round(t1-t0,1)})
        print(f"ep{ep:02d} | tr_loss {tr_loss:.4f} acc {tr_acc:.3f} | va_loss {va_loss:.4f} acc {va_acc:.3f} | {t1-t0:.1f}s")
    return hist

device='cuda' if torch.cuda.is_available() else 'cpu'