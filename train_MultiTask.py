import config
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import random
import numpy as np
from tqdm import tqdm
import warnings
# 忽略所有警告
warnings.filterwarnings('ignore')

from Loss_fun import *
from Utils import *
from DataSet.Multi_Dataset import *
from Models.MultiTaskModel import *


def train(args, model, optimizer, train_loader, loss_fn, device):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader):
        if len(args.input_img) > 1:
            input = [x.to(device) for x in imgs]
        else:
            input = imgs[0].to(device)
        labels = [x.to(device) for x in labels]
        outputs = model(input)
        loss = 0
        for i, ouput in enumerate(outputs):
            loss += loss_fn.get_loss(ouput, labels[i])

        # Backward and optimize
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播
        optimizer.step() # 更新参数
        total_loss += loss.item()*imgs[0].size(0)/len(args.label_names)
    epoch_loss = total_loss / len(train_loader.dataset)
    return epoch_loss

def val(args, epoch, model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = [[] for _ in range(len(args.label_names))]
    all_labels = [[] for _ in range(len(args.label_names))]
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            if len(args.input_img) > 1:
                input = [x.to(device) for x in imgs]
            else:
                input = imgs[0].to(device)
            labels = [x.to(device) for x in labels]
            outputs = model(input)
            loss = 0
            for i, output in enumerate(outputs):
                _, preds = torch.max(output, 1)
                all_preds[i].extend(preds.cpu().numpy())
                all_labels[i].extend(labels[i].cpu().numpy())
                loss += loss_fn.get_loss(output, labels[i])
                # loss += loss_fn(output, labels[i])
            total_loss += loss.item()*imgs[0].size(0)/len(args.label_names)
    epoch_loss = total_loss / len(val_loader.dataset)
    print_metrics_MultiTask(all_labels, all_preds, args.label_names)
    for i, labels in enumerate(all_labels):
        print(f'Task {args.label_names[i]}:')
        print(f'preds: {all_preds[i]}')
        print(f'label: {labels}')
    return epoch_loss

def test(args, model, test_loader, device):
    model.eval()
    total_loss = 0
    all_preds = [[] for _ in range(len(args.label_names))]
    all_labels = [[] for _ in range(len(args.label_names))]
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader):
            if len(args.input_img) > 1:
                input = [x.to(device) for x in imgs]
            else:
                input = imgs[0].to(device)
            labels = [x.to(device) for x in labels]
            outputs = model(input)
            loss = 0
            for i, output in enumerate(outputs):
                _, preds = torch.max(output, 1)
                all_preds[i].extend(preds.cpu().numpy())
                all_labels[i].extend(labels[i].cpu().numpy())
                loss += loss_fn.get_loss(output, labels[i])
            total_loss += loss.item()*imgs[0].size(0)/len(args.label_names)
    epoch_loss = total_loss / len(val_loader.dataset)
    print_metrics_MultiTask(all_preds, all_labels, args.label_names)
    for i, labels in enumerate(all_labels):
        print(f'Task {args.label_names[i]}:')
        print(f'preds: {all_preds[i]}')
        print(f'label: {labels}')
    return epoch_loss

if __name__ == '__main__':
    args = config.args
    seed_everything(args.seed)

    # 模型日志存储路径
    args.save_path = os.path.join(args.save_path,args.task_name,'seed%d_%s_lr%f_batch%d_epo%d_loss%s_opt%s_RLR%s_shape%s'%(args.seed, args.model_name, args.lr, args.batch_size, args.epochs, args.loss_func, args.opt,args.ReduceLR, args.in_size))
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)

    save_args(args.save_path, args)

    args.in_size = [int(x) for x in args.in_size.split(',')] # 输入图片限制size
    args.label_names = args.label_names.split(',') # 训练标签
    args.num_classes_list = [int(x) for x in args.num_classes_list.split(',')] # 训练标签
    args.input_img = [x for x in args.input_img.split(',')] # 输入图片类型

    device = torch.device('cuda' if args.gpu_num>0 else 'cpu')
    print(args.input_img)
    if len(args.input_img) > 1: 
        assert 'MultiInput' # 预留多输入模型
    else:
        model = MultiTaskResNet(args.num_classes_list).to(device)
    
    if args.gpu_num > 1:
        model = nn.DataParallel(model, device_ids=range(args.gpu_num)) # 多GPU训练
    
    loss_fn = Loss_fun(args, device) # 损失函数
    # loss_fn = MultiTaskLoss(args.num_classes_list, device) # 损失函数
    optimizer = set_optimizer(args, model) # 优化器
    scheduler = set_scheduler(args, optimizer) # 学习率调整器

    # 数据加载
    train_loader = DataLoader(dataset=ImageFolder(args, 'train'), batch_size=args.batch_size*args.gpu_num, num_workers=args.num_workers,shuffle=True)
    # val_loader = DataLoader(dataset=ImageFolder(args,'val'), batch_size=args.batch_size*args.gpu_num, num_workers=1,shuffle=False)
    test_loader = DataLoader(dataset=ImageFolder(args,'test'), batch_size=args.batch_size*args.gpu_num, num_workers=1,shuffle=False)

    print('train number:', len(train_loader.dataset))
    print('val number:', len(test_loader.dataset))

    best_loss = 10000
    for epoch in range(args.epochs):
        # if epoch == 50:
        #     for param in model.base_model.parameters():
        #         param.requires_grad = True  
        #     optimizer = set_optimizer(args, model) # 优化器  # 更新优化器以包含所有参数
        print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        train_loss = train(args,model, optimizer,train_loader,loss_fn,device)
        scheduler.step(train_loss)
        # print(f'train: Epoch {args.epochs}/{epoch - 1}, Loss: {train_loss:.4f}')

        val_loss = val(args, epoch, model, test_loader, loss_fn, device)
        print(f'val: Epoch {args.epochs}/{epoch - 1}, Loss: {val_loss:.4f}')

        state = {'net': model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(args.save_path, 'latent.pth'))
        if val_loss<best_loss:
            torch.save(state, os.path.join(args.save_path, 'best.pth'))
        best_loss = min(best_loss, val_loss)

    # print("===============TEST===============")    
    # test(args, model, test_loader, device)







    





