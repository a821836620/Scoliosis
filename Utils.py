import os
import csv
import random
import torchvision.models as models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from sklearn import manifold
import timm
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score


import matplotlib.pyplot as plt
from Models import *

from Loss_fun import *



# def get_Loss(args, device):
#     loss_fn = None
#     if args.loss_func == 'FocalLoss':
#         loss_fn = FocalLoss(device = device, gamma = 2.0).to(device)
#     elif args.loss_func == 'BCE':
#         loss_fn = nn.BCEWithLogitsLoss().to(device)
#     elif args.loss_func == 'CE':
#         loss_fn = nn.CrossEntropyLoss().to(device)
#     else:
#         assert False
#     return loss_fn

def get_model(model_name, label_len,pretrained=False):
    model = None
    if model_name == 'swintransformer':
        model = timm.create_model('swin_transformer_base_patch4_window7_224', pretrained=pretrained)
        num_features = model.head.in_features
        model.head = nn.Linear(num_features, label_len)

    elif model_name == 'densenet':
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, label_len)
    elif model_name == 'AlexNet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216,out_features=4096,bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=label_len)
        )
    elif model_name == 'Inception':
        model = models.inception_v3(pretrained=pretrained)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=label_len)
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=label_len, bias=True)
        )
    elif model_name == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        model.fc = nn.Linear(in_features, label_len) # 15 output classes

    elif model_name == 'ResNet34':
        model = models.resnet34(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        model.fc = nn.Linear(in_features, label_len) # 15 output classes  

    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        model.fc = nn.Linear(in_features, label_len) # 15 output classes  
    
    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, label_len) # 15 output classes  

    elif model_name == 'swin_transformer': 
        model = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)

        num_input = model.head.in_features
        model.head = nn.Linear(num_input, label_len)
        head_params = list(map(id, model.head.parameters()))
        base_params = filter(lambda p: id(p) not in head_params, model.parameters())
        params = [{"params":base_params, "lr":0.005},
                  {"params":model.head.parameters(), "lr": 0.05 },]
    elif model_name == 'SKNet':
        model = SKNet(label_len)
    
    return model

def set_scheduler(args, optimizer):
    # ReduceLR
    if args.ReduceLR == 'Plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7,
                                                         min_lr=10e-30)  # 测量损失动态下降学习率
    elif args.ReduceLR == 'LambdaLR':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1),
                                                last_epoch=-1)  # 根据epoch下降学习率
    elif args.ReduceLR == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 每训练step_size 个epoch 按 gamma倍数下降学习率
    elif args.ReduceLR == 'MutiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 7],
                                                   gamma=0.1)  # 每次遇到milestones里面的epoch 按gamma倍数更新学习率
    elif args.ReduceLR == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, last_epoch=-1)
    else:
        assert 'ReduceLR error'
    return scheduler

def get_confusion_matrix(trues, preds):
    conf_matrix = confusion_matrix(trues, preds)
    return conf_matrix

def plot_confusion_matrix(args, conf_matrix,epoch):
    if not os.path.exists(args.save_path+'/confusion_matrix'): os.makedirs(args.save_path+'/confusion_matrix')
    plt.figure()  # 创建一个新的图形对象
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    labels = args.train_label
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig(args.save_path+'/confusion_matrix/heatmap_confusion_matrix%d.jpg'%epoch)
    # plt.show()

def set_optimizer(args, model):
    # optimizer
    if args.opt == 'Adam':
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # 优化器 前期冻结ResNet只进行最后一层优化 
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.opt == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    return optimizer

def print_args(args):
    for arg, val in sorted(vars(args).items()):
        print('%s:%s'%(str(arg),str(val)))

def save_args(save_path, args):
    with open(os.path.join(save_path,'args.txt'),'w') as f:
        for arg, val in sorted(vars(args).items()):
            f.writelines('%s:%s\n'%(str(arg),str(val)))


def get_tsne(data, n_components = 2, n_images = None):
    if n_images is not None:
        data = data[:n_images]
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    return tsne_data

def plot_representations(train_label, data, labels, epoch, save_path, n_images = None,):
            
    if n_images is not None:
        data = data[:n_images]
        labels = labels[:n_images]
                
    # fig = plt.figure(figsize = (10, 10),dpi=600)
    fig = plt.figure(figsize = (15, 15), dpi=600)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, cmap = 'rainbow')
    a,b=scatter.legend_elements()
    b = []
    for label in train_label:
        b.append('$\\mathdefault{%s}$'%label)
    legend1 = ax.legend(a,b,loc="upper right", title="Classes")
    ax.add_artist(legend1)
    fig.savefig(save_path+'/%d.svg'%epoch)

def get_Data_label(args, splits = ['train']):
    imgs = os.listdir(os.path.join(args.root, 'Image_data', args.input_img[0])) # 获取输入图像文件夹下
    labels = []
    for split in splits:
        with open(os.path.join(args.root, args.task, split+'.csv'),'r') as file:
            reader = csv.reader(file)
            for line in reader:
                if line[0]+'.jpg' not in imgs: continue
                if int(line[1])>= args.num_classes: continue
                labels.append(int(line[1]))
    return labels

def get_Sampler(targets):
    # 获得数据集中不同label样本分布与比重

    # 计算整体不均衡权重
    class_counts = np.bincount(targets) # 统计每个类别的样本数
    class_weights = 1. / class_counts
    samples_weights = class_weights[targets]

    # 使用WeightedRandomSampler处理不均衡数据
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    return sampler, class_weights

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def print_metrics(labels, preds): #print_metrics(labels, preds, , preds_two)
    print('accuracy:',accuracy_score(labels, preds)) # 输出准确率
    print('precision:',precision_score(labels, preds, average='macro')) # 输出每个类别的精确度
    print('recall:',recall_score(labels, preds, average='macro')) # 输出每个类别的召回率
    print('f1:',f1_score(labels, preds, average='macro')) # 输出每个类别的F1值
    print('classification_report:\n',classification_report(labels, preds)) # 输出每个类别的精确度 召回率 F1值
    
    # print('roc_auc_score:',roc_auc_score(labels, preds, multi_class='ovr', average='macro')) # 输出每个类别的roc_auc值
    
    '''
    if len(set(labels))>1:
        print('roc_auc_score:',roc_auc_score(labels, preds.reshape(-1, 1), multi_class='ovr', average='weighted')) # 输出每个类别的roc_auc值
    else:
        print('ROC AUC Score: Not defined (only one class present in y_true)') # 输出roc_auc值不可用的提示
    '''

    '''
    auc = roc_auc_score(labels, preds_two, multi_class='ovr', average='weighted')
    print('roc_auc_score:',auc) # 输出每个类别的roc_auc值
    cm = confusion_matrix(labels, preds) # 输出混淆矩阵
    print('Confusion Matrix:\n', cm)
    return auc
    '''

def print_metrics_MultiTask(labels, preds, Tasks): # 多任务分类指标
    for i, task in enumerate(Tasks):
        print('task:',task)
        print_metrics(labels[i], preds[i])