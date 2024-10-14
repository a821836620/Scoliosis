import torch.nn.functional as F
import torch
import torch.nn as nn

# class FocalLoss(nn.Module):

#     def __init__(self, device, gamma=1.0):
#         super(FocalLoss, self).__init__()
#         self.device = device
#         self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
#         self.eps = 1e-6

#     #         self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)

#     def forward(self, input, target):
#         BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean').to(self.device)
#         #         BCE_loss = self.BCE_loss(input, target)
#         pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
#         F_loss = (1 - pt) ** self.gamma * BCE_loss
#         return F_loss.mean()

class Loss_fun():
    def __init__(self, args, device, class_weights=None):
        self.num_classes = args.num_classes
        self.device = device
        self.loss_func = args.loss_func
        class_weights_tensor = None
        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        if args.loss_func == 'FocalLoss':
            weight = torch.tensor(class_weights/class_weights.sum()).to(device) if class_weights is not None else None
            self.loss_fn = FocalLoss(weight=weight).to(device)
        elif args.loss_func == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights_tensor).to(device)
        elif args.loss_func == 'CE':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
        else:
            assert False
            
    def get_loss(self, output, target):
        if self.loss_func in ['CE','FocalLoss']:
            return self.loss_fn(output, target)
        else:
            target = F.one_hot(target, self.num_classes).to(self.device).float()
            return self.loss_fn(output, target)
         

class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight.float() if weight is not None else None

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes_list, device, gamma=1.0):
        super(MultiTaskLoss, self).__init__()
        self.device = device
        self.num_classes_list = num_classes_list
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
        self.eps = 1e-6
        self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self, outputs, labels):
        loss = 0
        for i, output in enumerate(outputs):
            BCE_loss = self.BCE_loss(output, labels[i])
            pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
            F_loss = (1 - pt) ** self.gamma * BCE_loss
            loss += F_loss.mean()
        return loss
