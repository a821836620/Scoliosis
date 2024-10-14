import argparse

parser = argparse.ArgumentParser(description='scoliosis predict')

parser.add_argument('--task_name', type=str, default='Debug')
parser.add_argument('--save_path', type=str, default='/home/hjz/Exp/scoliosis', help='Exp save path about train para, model and result')
parser.add_argument('--input_img', type=str, default='CT_front', help='front left right back CT_front CT_side')


# training 
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--root', default='/home/hjz/data/scoliosis/clean_new/', help='trainset root path')

parser.add_argument('--model_name', type=str, default='ResNet', help='densenet AlexNet vgg ResNet')
parser.add_argument('--lr', type=float, default=0.1,help='learning rate')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of training')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--loss_func', type=str, default='CE', help='FocalLoss BCE CE')
parser.add_argument('--opt', type=str, default='Adam', help='Adam SGD RMSprop')
parser.add_argument('--ReduceLR', type=str, default='Plateau', help='Plateau, StepLR')
parser.add_argument('--in_size', default='512,410', type=str, help='input data size')
parser.add_argument('--num_classes', type=int,default=7,help='number of classes')


parser.add_argument('--gpu_num', type=int, default=1, help='number of gpu to train')
parser.add_argument('--num_workers', default=8, type=int)

# test
parser.add_argument('--test_path', type=str, default='/home/hjz/work/multiClass/experiments/yanhong_c10/seed1234_densenet_lr0.060000_b32_epo500_lossCE_optAdam_RLRPlateau_shape256')
parser.add_argument('--test_model', type=str, default='best.pth')

# MultiTask
parser.add_argument('--label_names', type=str, default='T,L,TL,Type,Cobb', help='label names')
parser.add_argument('--num_classes_list', type=str, default='5,3,5,5,4', help='number of classes')

# Sample Task
parser.add_argument('--task', type=str, default='Cobb', help='T L TL Type Cobb All')
parser.add_argument('--train_label', type=str, default='0-5,6-10,11-15,16-20,21-25', help='')

args = parser.parse_args()