import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import numpy as np
from tqdm import tqdm

import config
from Utils import *
from DataSet.Dataset import *
from Matrix import *
from Models.Model import *

import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def train(args, model, optimizer,train_loader,loss_fn,device):
    model.train()
    total_loss = 0
    for imgs, label in tqdm(train_loader):
        if len(args.input_img) > 1:
            input = [x.to(device) for x in imgs]
        else:
            input = imgs[0].to(device)
        label = label.to(device).float()

        outputs = model(input)

        loss = loss_fn.get_loss(outputs, label.long()) # 计算损失
        # Backward and optimize
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        total_loss += loss.item()*imgs[0].size(0)
    
    epoch_loss = total_loss / len(train_loader.dataset)
    return epoch_loss


def val(args, epoch, model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
   
    preds = []
    labels = []
    preds_two = []

    with torch.no_grad():
        for imgs, label in tqdm(val_loader):
            if len(args.input_img) > 1:
                input = [x.to(device) for x in imgs]
            else:
                input = imgs[0].to(device)

            label = label.to(device).float()
            outputs = model(input)

            loss = loss_fn.get_loss(outputs, label.long())
            total_loss += loss.item()*imgs[0].shape[0]

            preds.extend(torch.argmax(F.softmax(outputs.cpu(), dim=1), dim=1).numpy())
            preds_two.extend(F.softmax(outputs.cpu(), dim=1).numpy())   
            labels.extend(label.cpu().numpy())
    
    epoch_loss = total_loss / len(val_loader.dataset)
    auc = print_metrics(labels, preds, preds_two)
    print('preds:',preds)
    print('labels:',labels)
        
    return epoch_loss, auc

def test(args, model, test_loader, device):
    model.eval()
    total_loss = 0
   
    preds = []
    labels = []
    print("===============TEST===============")
    with torch.no_grad():
        for imgs, label in tqdm(test_loader):
            if len(args.input_img) > 1:
                input = [x.to(device) for x in imgs]
            else:
                input = imgs[0].to(device)

            label = label.to(device).float()
            outputs = model(input)

            preds.extend(torch.argmax(F.softmax(outputs.cpu(), dim=1), dim=1).numpy())   
            labels.extend(label.cpu().numpy())
    
    
    print_metrics(preds, labels)
    print('preds:',preds)
    print('labels:',labels)


if __name__ == '__main__':
    args = config.args
    seed_everything(args.seed)
    
    args.save_path = os.path.join(args.save_path,args.task_name,args.task,'seed%d_%s_lr%f_batch%d_epo%d_loss%s_opt%s_RLR%s_shape%s'%(args.seed, args.model_name, args.lr, args.batch_size, args.epochs, args.loss_func, args.opt,
    args.ReduceLR, args.in_size))
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
    save_args(args.save_path, args)
    print_args(args)

    args.in_size = [int(x) for x in args.in_size.split(',')]
    args.train_label = args.train_label.split(',')
    args.num_classes = len(args.train_label)
    args.input_img = [x for x in args.input_img.split(',')]

    device = torch.device('cuda' if args.gpu_num>0 else 'cpu')

    print(args.input_img)
    print(args.task)

    if len(args.input_img) > 1:
        model = MultiInput(args, args.num_classes).to(device)
    else:
        model = get_model(args.model_name, args.num_classes).to(device)
    

    if args.gpu_num > 1:
        model = torch.nn.DataParallel(model)  # multi-GPU

    labels = get_Data_label(args) #
    train_sampler, class_weights = get_Sampler(labels)

    # ImageFolder调用read_data_paths 读train.csv, val.csv
     # 数据加载 只读了train和val
    # train_loader = DataLoader(dataset=ImageFolder(args, 'train'), batch_size=args.batch_size*args.gpu_num, sampler=train_sampler, num_workers=args.num_workers)
    train_loader = DataLoader(dataset=ImageFolder(args, 'train'), batch_size=args.batch_size*args.gpu_num, num_workers=args.num_workers,shuffle=True )
    val_loader = DataLoader(dataset=ImageFolder(args,'val'), batch_size=args.batch_size*args.gpu_num, num_workers=args.num_workers,shuffle=False)    
    test_loader = DataLoader(dataset=ImageFolder(args,'test'), batch_size=args.batch_size*args.gpu_num, num_workers=args.num_workers,shuffle=False)

    # loss_fn = Loss_fun(args, device, class_weights)
    loss_fn = Loss_fun(args, device)
    optimizer = set_optimizer(args, model)
    scheduler = set_scheduler(args, optimizer)

    print('train number:', len(train_loader.dataset))
    print('val number:', len(val_loader.dataset))

    best_auc = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss = train(args,model, optimizer,train_loader,loss_fn,device)
        scheduler.step(train_loss)
        print("train: epoch {:4d}, loss {:.4f}".format(epoch, train_loss))

        val_loss_list, auc = val(args, epoch, model, val_loader, loss_fn, device)
        print("val: epoch {:4d}, loss {:.4f}".format(epoch, val_loss_list))

        state = {'net': model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(args.save_path, 'latent.pth'))

        if epoch>1: # 样本量很小 可能早期就能达到最好效果，后期可能过拟合
            # if val_loss_list<best_loss:
            if auc > best_auc: # 修改为通过auc保存最好的模型
                torch.save(state, os.path.join(args.save_path, 'best.pth'))
                best_epoch = epoch
                best_auc = auc
        print('best epoch:', best_epoch, 'best loss:', best_auc)
    
    # 测试前加载最佳模型
    best_state = torch.load(os.path.join(args.save_path, 'best.pth'))
    model.load_state_dict(best_state['net'])
    
    test(args, model, test_loader, device)


