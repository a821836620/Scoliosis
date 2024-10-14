import os
import csv
import cv2
import torch.utils.data as data
from torchvision import transforms
from DataSet.Utils import ResizeWithPadding
def Get_data(args, val = False): # val = True: return train and val data, val = False: return train and test data
    files = []
    with open(os.path.join(args.root, 'train.csv'),'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if int(line[1])>= args.num_classes: continue
            image_path = []
            for input_img in args.input_img:
                image_path.append(os.path.join(args.root, input_img, line[0]+'.jpg'))
            files.append({
                "image_path": image_path,
                "label":int(line[1])
            })

    with open(os.path.join(args.root, 'val.csv'),'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if int(line[1])>= args.num_classes: continue
            image_path = []
            for input_img in args.input_img:
                image_path.append(os.path.join(args.root, input_img, line[0]+'.jpg'))
            files.append({
                "image_path": image_path,
                "label":int(line[1])
            })

    files_test = []
    with open(os.path.join(args.root, 'test.csv'),'r') as file:
        reader = csv.reader(file)
        for line in reader:
            if int(line[1])>= args.num_classes: continue
            image_path = []
            for input_img in args.input_img:
                image_path.append(os.path.join(args.root, input_img, line[0]+'.jpg'))
            files_test.append({
                "image_path": image_path,
                "label":int(line[1])
            })
    
    return files, files_test

class ImageFolder(data.Dataset):
    def __init__(self, args, files, split):
        self.files = files
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                ResizeWithPadding(args.in_size),
                # transforms.Resize((args.in_size[0],args.in_size[1])),
                #transforms.RandomHorizontalFlip(p=0.3),  # 50% 概率水平翻转
                #transforms.RandomRotation(10),  # 随机旋转±15度，概率为100%
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 随机颜色变化
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                ResizeWithPadding(args.in_size),
                # transforms.Resize((args.in_size[0],args.in_size[1])),
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