import os
import csv
import cv2
import torch.utils.data as data
from torchvision import transforms
from DataSet.Utils import ResizeWithPadding

def read_data_paths(root,task, input_imgs, split, num_classes):
    files = []
    imgs = {}
    for i, input_img in enumerate(input_imgs):
        imgs[input_img] = os.listdir(os.path.join(root, input_img)) # 获取输入图像文件夹下
    with open(os.path.join(root,task,split+'.csv'),'r') as file:
        reader = csv.reader(file)
        for line in reader:
            flag = 0
            if int(line[1])>= num_classes: continue
            image_path = []
            for input_img in input_imgs:
                if line[0]+'.jpg' not in imgs[input_img]:
                    flag = 1
                    break
                image_path.append(os.path.join(root, input_img, line[0]+'.jpg'))
            if flag == 1: continue # 图片不存在 跳过
            files.append({
                "image_path": image_path,
                "label":int(line[1])
            })
    return files

class ImageFolder(data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.root = args.root
        self.num_classes = args.num_classes
        self.files = read_data_paths(self.root, args.task, args.input_img, split, args.num_classes)
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize((args.in_size[0],args.in_size[1])),
                ResizeWithPadding(args.in_size),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色变化
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                # transforms.Resize((args.in_size[0],args.in_size[1])),
                ResizeWithPadding(args.in_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    
    def __getitem__(self, index):
        file = self.files[index]
        imgs = []
        for img_path in file['image_path']:
            img = cv2.imread(img_path)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)
        return imgs, file['label']
    
    def __len__(self):
        return len(self.files)
    
