import os
import shutil
import random
from tqdm import tqdm
# import xlrd
import csv
import re
import math
import cv2 as cv
import xlrd
from pypinyin import pinyin, Style
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

def get_label_weight():
    path = '/home/hjz/data/scoliosis/org/影像数据_2.xlsx'
    target = '/home/hjz/data/scoliosis/clean_new'

    wb = xlrd.open_workbook(path)
    sheet1 = wb.sheets()[1]
    nrows = sheet1.nrows
    print(sheet1.cell(0,0).value, sheet1.cell(0,7).value)
    level_map = {}
    dim = {0:[],1:[],2:[],3:[],4:[]}
    for r in range(1,nrows):
        if sheet1.cell(r,7).value[0:2] == '0-':
            level_map[sheet1.cell(r,0).value] = 0
            dim[0].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '5-':
            level_map[sheet1.cell(r,0).value] = 1
            dim[1].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '11':
            level_map[sheet1.cell(r,0).value] = 2
            dim[2].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '16':
            level_map[sheet1.cell(r,0).value] = 3
            dim[3].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '21':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '20':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)    
        elif sheet1.cell(r,7).value[0:2] == '26':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '31':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)
        else:
            print(sheet1.cell(r,0).value, sheet1.cell(r,7).value)

    all_train = []
    all_val = []
    all_test = []
    for key in dim.keys():
        print(key, dim[key], len(dim[key]))
        train_ID = random.sample(dim[key], int(len(dim[key])*7/10))
        val_test_ID = [x for x in dim[key] if x not in train_ID]
        val_ID = random.sample(val_test_ID, int(len(val_test_ID)/3))
        test_ID = [x for x in val_test_ID if x not in val_ID]
        all_train.extend(train_ID)
        all_val.extend(val_ID)
        all_test.extend(test_ID)
    print(len(all_train), len(all_val), len(all_test))
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)

    with open(os.path.join(target, 'train.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in all_train:
            writer.writerow([key,level_map[key]])
    
    with open(os.path.join(target, 'val.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in all_val:
            writer.writerow([key,level_map[key]])
    
    with open(os.path.join(target, 'test.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in all_test:
            writer.writerow([key,level_map[key]])


def get_label():
    path = '/home2/hjz_data/scoliosis/org/影像数据.xlsx'
    target = '/home2/hjz_data/scoliosis/clean/'

    wb = xlrd.open_workbook(path)
    sheet1 = wb.sheets()[1]
    nrows = sheet1.nrows
    print(sheet1.cell(0,0).value, sheet1.cell(0,7).value)
    level_map = {}
    dim = {0:[],1:[],2:[],3:[],4:[],5:[]}
    for r in range(1,nrows):
        if sheet1.cell(r,7).value[0:2] == '0-':
            level_map[sheet1.cell(r,0).value] = 0
            dim[0].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '5-':
            level_map[sheet1.cell(r,0).value] = 1
            dim[1].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '11':
            level_map[sheet1.cell(r,0).value] = 2
            dim[2].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '16':
            level_map[sheet1.cell(r,0).value] = 3
            dim[3].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '21':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '26':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)
        elif sheet1.cell(r,7).value[0:2] == '31':
            level_map[sheet1.cell(r,0).value] = 4
            dim[4].append(sheet1.cell(r,0).value)
        else:
            print(sheet1.cell(r,0).value, sheet1.cell(r,7).value)
    
    # with open(target, 'w') as file:
    #     writer = csv.writer(file)
    #     for key in level_map.keys():
    #         writer.writerow([key,level_map[key]])
    Id = level_map.keys()
    train_ID = random.sample(Id, int(len(Id)*8/10))
    val_test_ID = [x for x in Id if x not in train_ID]
    val_ID = random.sample(val_test_ID, int(len(val_test_ID)/2))
    test_ID = [x for x in val_test_ID if x not in val_ID]
    random.shuffle(train_ID)
    random.shuffle(val_ID)
    random.shuffle(test_ID)


    with open(os.path.join(target, 'train.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in train_ID:
            writer.writerow([key,level_map[key]])
    
    with open(os.path.join(target, 'val.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in val_ID:
            writer.writerow([key,level_map[key]])
    
    with open(os.path.join(target, 'test.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in test_ID:
            writer.writerow([key,level_map[key]])


        # for c in range(ncols):
        #     print(sheet1.cell(r, c).value, end = ' ')
        # print()

def rename(path, cases):
    for case in tqdm(cases):
        new_case = remove_chinese(case)
        shutil.move(os.path.join(path, case), os.path.join(path, new_case))

def org_clean():
    '''
    整理原始数据，根据位置重命名
    '''
    path = '/home2/hjz_data/GU/org/原始图像/'
    xlsx_path = '/home2/hjz_data/GU/org/影像数据 1120 数据整理(1).xlsx'
    target = '/home2/hjz_data/GU/clean/'

    # back  CT_front  CT_side  front  left  right
    if not os.path.exists(os.path.join(target, 'front')): os.makedirs(os.path.join(target, 'front'))
    if not os.path.exists(os.path.join(target, 'left')): os.makedirs(os.path.join(target, 'left'))
    if not os.path.exists(os.path.join(target, 'back')): os.makedirs(os.path.join(target, 'back'))
    if not os.path.exists(os.path.join(target, 'right')): os.makedirs(os.path.join(target, 'right'))
    if not os.path.exists(os.path.join(target, 'CT_front')): os.makedirs(os.path.join(target, 'CT_front'))
    if not os.path.exists(os.path.join(target, 'CT_side')): os.makedirs(os.path.join(target, 'CT_side'))
    cases = os.listdir(os.path.join(target, 'front'))
    rename(os.path.join(target, 'front'), cases)
    
    cases = os.listdir(os.path.join(target, 'left'))
    rename(os.path.join(target, 'left'), cases)

    cases = os.listdir(os.path.join(target, 'back'))
    rename(os.path.join(target, 'back'), cases)

    cases = os.listdir(os.path.join(target, 'right'))
    rename(os.path.join(target, 'right'), cases)

    cases = os.listdir(os.path.join(target, 'CT_front'))
    rename(os.path.join(target, 'CT_front'), cases)

    cases = os.listdir(os.path.join(target, 'CT_side'))
    rename(os.path.join(target, 'CT_side'), cases)
    
    # cases = os.listdir(path)
    # for case in tqdm(cases):
    #     imgs = os.listdir(os.path.join(path, case))
    #     if imgs != []:
    #         print(case)

        # camera =[x for x in imgs if x[0:4] == 'IMG_']
        # CT = [x for x in imgs if x[0:4] != 'IMG_']

        # camera = sorted(camera)
        # if len(camera) != 4:
        #     print(case, 'camera:'+str(len(camera)))
        # else:
        #     shutil.move(os.path.join(path,case,camera[0]), os.path.join(target, 'front',case+'.jpg'))
        #     shutil.move(os.path.join(path,case,camera[1]), os.path.join(target, 'left',case+'.jpg'))
        #     shutil.move(os.path.join(path,case,camera[2]), os.path.join(target, 'back',case+'.jpg'))
        #     shutil.move(os.path.join(path,case,camera[3]), os.path.join(target, 'right',case+'.jpg'))

        # CT = sorted(CT)
        # if len(CT) != 2:
        #     print(case, 'CT:'+str(len(CT)))
        #     continue
        # shutil.move(os.path.join(path,case,CT[0]), os.path.join(target, 'CT_front', case+'.jpg'))
        # shutil.move(os.path.join(path,case,CT[1]), os.path.join(target, 'CT_side', case+'.jpg'))
            # print(CT)


        # shutil.copy()
        
        # for img in imgs: 
    
def swap():
    path1 = '/home2/hjz_data/scoliosis/clean/CT_front/'
    path2 = '/home2/hjz_data/scoliosis/clean/CT_side/'
    swap_list = ['N05','N08','N09','N13','N14','N15','N16','N17','N19','N29','N25','N22','N21','N20','S04','N230','N206','S07','S08','S46','S47','S48','S52','S53','S59','S60','S63']
    # for swap_id in tqdm(swap_list):
    #     shutil.move(os.path.join(path1,swap_id+'.jpg'), os.path.join(path1,'tmp.jpg'))
    #     shutil.move(os.path.join(path2,swap_id+'.jpg'), os.path.join(path1,swap_id+'.jpg'))
    #     shutil.move(os.path.join(path1,'tmp.jpg'), os.path.join(path2,swap_id+'.jpg'))
    # for id in tqdm(range(12,27)):
    #     swap_id = 'S'+str(id)
    #     shutil.move(os.path.join(path1,swap_id+'.jpg'), os.path.join(path1,'tmp.jpg'))
    #     shutil.move(os.path.join(path2,swap_id+'.jpg'), os.path.join(path1,swap_id+'.jpg'))
    #     shutil.move(os.path.join(path1,'tmp.jpg'), os.path.join(path2,swap_id+'.jpg'))
    
    # for id in range(28,34):
    #     swap_id = 'S'+str(id)
    #     shutil.move(os.path.join(path1,swap_id+'.jpg'), os.path.join(path1,'tmp.jpg'))
    #     shutil.move(os.path.join(path2,swap_id+'.jpg'), os.path.join(path1,swap_id+'.jpg'))
    #     shutil.move(os.path.join(path1,'tmp.jpg'), os.path.join(path2,swap_id+'.jpg'))
    
    for id in range(39,43):
        swap_id = 'S'+str(id)
        shutil.move(os.path.join(path1,swap_id+'.jpg'), os.path.join(path1,'tmp.jpg'))
        shutil.move(os.path.join(path2,swap_id+'.jpg'), os.path.join(path1,swap_id+'.jpg'))
        shutil.move(os.path.join(path1,'tmp.jpg'), os.path.join(path2,swap_id+'.jpg'))



def remove_chinese(filename):
    chinese = r'[\u4e00-\u9fa5]'
    new_filename = re.sub(chinese, '', filename)
    return new_filename

def put_class():
    path = '/home2/hjz_data/scoliosis/org/影像数据.xlsx'
    wb = xlrd.open_workbook(path)
    sheet1 = wb.sheets()[1]
    nrows = sheet1.nrows
    print(sheet1.cell(0,0).value, sheet1.cell(0,7).value)
    level_map = {}
    level_num = [0]*7
    for r in range(1,nrows):
        if sheet1.cell(r,7).value[0:2] == '0-':
            level_map[sheet1.cell(r,0).value] = 0
            level_num[0] += 1
        elif sheet1.cell(r,7).value[0:2] == '5-':
            level_map[sheet1.cell(r,0).value] = 1
            level_num[1] += 1
        elif sheet1.cell(r,7).value[0:2] == '11':
            level_map[sheet1.cell(r,0).value] = 2
            level_num[2] += 1
        elif sheet1.cell(r,7).value[0:2] == '16':
            level_map[sheet1.cell(r,0).value] = 3
            level_num[3] += 1
        elif sheet1.cell(r,7).value[0:2] == '21':
            level_map[sheet1.cell(r,0).value] = 4
            level_num[4] += 1
        elif sheet1.cell(r,7).value[0:2] == '26':
            level_map[sheet1.cell(r,0).value] = 5
            level_num[5] += 1
        elif sheet1.cell(r,7).value[0:2] == '31':
            level_map[sheet1.cell(r,0).value] = 6
            level_num[6] += 1
        else:
            print(sheet1.cell(r,0).value, sheet1.cell(r,7).value)
    print(level_num)
# put_class()
# swap()
# org_clean()
# get_label_weight()

def count_label():
    path = '/home/hjz/data/scoliosis/clean_new'
    fw = open(path+'/data.csv', 'r')
    label = fw.readlines()
    T_label_map = {}
    L_label_map = {}
    TL_label_map = {}
    Type_label_map = {}
    class_label_map = {}
    for l in label:
        l = l.strip().split(',')
        if l[1] in T_label_map.keys():
            T_label_map[l[1]] += 1
        else:
            T_label_map[l[1]] = 1
        
        if l[2] in L_label_map.keys():
            L_label_map[l[2]] += 1  
        else:
            L_label_map[l[2]] = 1

        if l[3] in TL_label_map.keys():
            TL_label_map[l[3]] += 1
        else:
            TL_label_map[l[3]] = 1

        if l[4] in Type_label_map.keys():
            Type_label_map[l[4]] += 1
        else:
            Type_label_map[l[4]] = 1
        
        if l[5] in class_label_map.keys():
            class_label_map[l[5]] += 1
        else:
            class_label_map[l[5]] = 1
    # print(label_map)
    print('T标签总数：%d 分布:'%len(T_label_map.keys()))
    sorted_keys = sorted(T_label_map.keys())
    for key in sorted_keys:
        print(key, T_label_map[key])

    print('L标签总数：%d 分布:'%len(L_label_map.keys()))
    sorted_keys = sorted(L_label_map.keys())
    for key in sorted_keys:
        print(key, L_label_map[key])

    print('TL标签总数：%d 分布:'%len(TL_label_map.keys()))
    sorted_keys = sorted(TL_label_map.keys())
    for key in sorted_keys:
        print(key, TL_label_map[key])

    print('分型标签总数：%d 分布:'%len(Type_label_map.keys()))
    sorted_keys = sorted(Type_label_map.keys())
    for key in sorted_keys:
        print(key, Type_label_map[key])

    print('分级标签总数：%d 分布:'%len(class_label_map.keys()))
    sorted_keys = sorted(class_label_map.keys())
    for key in sorted_keys:
        print(key, class_label_map[key])

    # T型分布: 1-19:29 20-25:31 26-29:28 30-34:31 36-50:26
    # L型分布: 1-39:6 40-45:24 46-50:52 51-55:45 56+:10
    # TL型分布: 1-29:8 30-34:34 35-40:64 41+:31
    # 分型3类别
    # 分级5类别    
def split_cl(num,type):
    if type is 'Cobb':
        types = ['0-4°','5-10°','11-15°','16-20°','21-25°','26-30','31°以上']
        if isinstance(num, str):
        # if '-' in num or '以上' in num:
            id = types.index(num)
            if id >=4:
                return 4
            else:
                return id
        else:
            num = int(num)
            if num <= 4:
                return 0
            elif num <= 10:
                return 1
            elif num <= 15:
                return 2
            elif num <= 20:
                return 3
            elif num <= 25:
                return 4
            else:
                return 5
            
    elif type is 'Type':
        if num[0] is 'N':
            return 0
        elif num[0] is 'C':
            return 1
        else:
            return 2

    elif type is 'T':
        if num <= 19:
            return 0
        elif num <= 25:
            return 1
        elif num <= 29:
            return 2
        elif num <= 35:
            return 3
        else:
            return 4
    
    if type is 'L':
        if num <= 39:
            return 0
        elif num <= 45:
            return 1
        elif num <= 50:
            return 2
        elif num <= 55:
            return 3
        else:
            return 4
        
    if type is 'TL':
        if num <= 29:
            return 0
        elif num <= 34:
            return 1
        elif num <= 40:
            return 2
        else:
            return 3

def main(Task = 'All'):
    path = '/home/hjz/data/scoliosis/clean_new'
    fr = open(path+'/data_new.csv', 'r')
    lines = fr.readlines()
    label_map = {}
    # print(lines[0])
    lines = lines[1:]
    for l in lines:
        l = l.strip().split(',')
        T = split_cl(float(l[1]),'T')
        L = split_cl(float(l[2]),'L')
        TL = split_cl(float(l[3]),'TL')
        Type = split_cl(l[4],'Type')
        class_ = int(l[5])
        label_map[l[0]] = [T,L,TL,Type,class_]
    
    if not os.path.exists(os.path.join(path, Task)): os.makedirs(os.path.join(path, Task))
    fw_train = open(os.path.join(path,Task,'train.csv'), 'w')
    fw_val = open(os.path.join(path,Task,'val.csv'), 'w')
    fw_test = open(os.path.join(path,Task,'test.csv'), 'w')

    if Task is 'All':
        key_list = list(label_map.keys())
        val_keys = random.sample(key_list, math.ceil(len(key_list)/10)) # 验证集
        train_test_keys = [x for x in key_list if x not in val_keys]
        test_keys = random.sample(train_test_keys, math.ceil(len(train_test_keys)*2/9)) # 测试集
        train_keys = [x for x in train_test_keys if x not in test_keys] # 训练集
        
        for key in train_keys:
            fw_train.write(key+','+str(label_map[key][0])+','+str(label_map[key][1])+','+str(label_map[key][2])+','+str(label_map[key][3])+','+str(label_map[key][4])+'\n')
        fw_train.close

        for key in val_keys:
            fw_val.write(key+','+str(label_map[key][0])+','+str(label_map[key][1])+','+str(label_map[key][2])+','+str(label_map[key][3])+','+str(label_map[key][4])+'\n')   
        fw_val.close

        for key in test_keys:
            fw_test.write(key+','+str(label_map[key][0])+','+str(label_map[key][1])+','+str(label_map[key][2])+','+str(label_map[key][3])+','+str(label_map[key][4])+'\n')
        fw_test.close
        return 

    elif Task is 'T':
        Task_id = 0
    elif Task is 'L':
        Task_id = 1
    elif Task is 'TL':
        Task_id = 2
    elif Task is 'Type':
        Task_id = 3
    elif Task is 'Cobb':
        Task_id = 4
    else:
        assert False, 'Task Error'

    label_list = set([label_map[x][Task_id] for x in label_map.keys()]) # 获取T标签列表
    all_train_id = []
    all_val_id = []
    all_test_id = []

    for lab  in label_list:
        id_list = [x for x in label_map.keys() if label_map[x][Task_id] == lab] # 获取到label为lab的id序列
        val_keys = random.sample(id_list, math.ceil(len(id_list)/10)) # 验证集
        train_test_keys = [x for x in id_list if x not in val_keys]
        test_keys = random.sample(train_test_keys, math.ceil(len(train_test_keys)*2/9)) # 测试集
        train_keys = [x for x in train_test_keys if x not in test_keys] # 训练集

        # 根据标签分布来划分训练集，验证集，测试集

        all_train_id.extend(train_keys)
        all_val_id.extend(val_keys)
        all_test_id.extend(test_keys)
    
    # 打乱数据
    random.shuffle(all_train_id)
    random.shuffle(all_val_id)
    random.shuffle(all_test_id)

    for key in all_train_id:
        fw_train.write(key+','+str(label_map[key][Task_id])+'\n')
    for key in all_val_id:
        fw_val.write(key+','+str(label_map[key][Task_id])+'\n')
    for key in all_test_id:
        fw_test.write(key+','+str(label_map[key][Task_id])+'\n')
    fw_train.close
    fw_val.close
    fw_test.close


# main('Type')
# label()
def data_clean():
    path = '/home/hjz/data/scoliosis/clean_new/CT_front'
    cases = os.listdir(path)
    max_high = 0
    max_scale = -1
    min_scale = 100
    for case in tqdm(cases):
        image = cv.imread(os.path.join(path,case))
        max_high = max(max_high, image.shape[0])
        max_scale = max(max_scale, image.shape[1]/image.shape[0])
        min_scale = min(min_scale, image.shape[1]/image.shape[0])
        if image.shape[1]/image.shape[0]>0.5:
            print(case)
    print(max_high, max_scale, min_scale)

# 影像数据 tabs=[1,0], cols = [[7,3],[3,4,5]]
# 20240715 tabs = [0], cols = [1,5,2,3,4]
def read_xlrx(path,lab_maps, tabs = [0], cols = [0],rename = []):
    
    wb = xlrd.open_workbook(path)
    for i, tab in enumerate(tabs):
        sheet = wb.sheets()[tab]
        nrows = sheet.nrows
        for r in range(1, nrows):
            key = sheet.cell(r, 0).value # 获取到ID
            if rename != []:
                pinyin_list = pinyin(key, style=Style.NORMAL)
                key_new = ''.join([item[0].capitalize() for item in pinyin_list])
                if str(key_new)+'.jpg' not in rename:
                    print(key)
                    continue
                else:
                    key = key_new
            lab_list = []
            for lab_idx in cols[i]: # 获取到标签
                lab_list.append(sheet.cell(r, lab_idx).value)
            if key not in lab_maps.keys():
                lab_maps[key] = lab_list
            else:
                lab_maps[key].extend(lab_list)
    return lab_maps

def random_split(label_counts, train_indices, test_indices, train_ratio = 0.7): # 根据第一个任务的每个标签类别按照比例划分数据集
    for label, samples in label_counts[0].items():
        random.shuffle(samples)
        split_point = int(len(samples) * train_ratio)
        train_indices.update(samples[:split_point])
        test_indices.update(samples[split_point:])


def split_data(label_counts, train_indices, test_indices, train_ratio = 0.7, max_iter=10):
    # 首先根据第一个任务随机划分数据集，然后看划分后的训练测试，在剩下4个任务上是否同时包含所有种类的标签，如果不包含则回到第一步，设置一个最大迭代次数
    Second_id = 0
    Sec_train_indices = set()
    Sec_test_indices = set()
    for _ in range(max_iter):
        random_split(label_counts, train_indices, test_indices, train_ratio)
        flag = 0 # 用来记录是否满足

        for i in range(1, len(label_counts)): # 看下面四个任务是否满足条件
            for label, samples in label_counts[i].items(): # 获取一个任务上某类标签的所有样本
                train_samples = [s for s in samples if s in train_indices] # 看这类标签在训练集上的数量
                test_samples = [s for s in samples if s in test_indices] # 看这类标签在测试集上的数量
                if len(train_samples)>0 and len(test_samples)>0: # 如果同时在训练和测试集上 继续判断下类标签
                    continue
                else: # 如果不同时在，则重新划分数据集
                    flag = 1
                    break
            if flag == 1: # 说明这类标签不满足条件，重新划分数据集
                if i > Second_id: # 记录下当前满足比较多任务的 训练测试集
                    Second_id = i
                    Sec_train_indices = train_indices
                    Sec_test_indices = test_indices
                break
        if flag == 0: # 所有任务都满足条件 则跳出循环
            break
    
    if flag == 1: # 说明没有满足条件的数据集
        train_indices = Sec_train_indices
        test_indices = Sec_test_indices
    

                
        


def split_train_test(lab_maps):
    # 统计每个类别的样本数量
    label_counts = [defaultdict(list) for _ in range(5)]

    # label_counts是个list 里面每个元素是一个map 代表示一个任务 里面的key是标签 value是样本id
    for sample_id, labels in lab_maps.items():
        for i, label in enumerate(labels):
            label_counts[i][label].append(sample_id) 
            

    # 按照优先级划分数据集
    train_indices = set()
    test_indices = set()

    split_data(label_counts, train_indices, test_indices)

 
    
    # 最终的训练集和测试集
    train_data = {sample_id: lab_maps[sample_id] for sample_id in train_indices}
    test_data = {sample_id: lab_maps[sample_id] for sample_id in test_indices}

    # 根据索引划分训练集和测试集
    return train_data, test_data


def Count_print(data, task_list):
    tj = {}
    for key, lab_list in data.items():
        for i, lab in enumerate(lab_list):
            if i not in tj.keys():
                tj[i] = {lab:1}
            elif lab not in tj[i].keys():
                tj[i][lab] = 1
            else:
                tj[i][lab] += 1
    
    for i, task in enumerate(task_list):
        print(task)
        for key in tj[i].keys():
            print(key, tj[i][key])
        print('-------------------------')

def Merge():
    path1 = '/home/hjz/data/scoliosis/clean_new/Image_data/CT_front1/'
    path2 = '/home/hjz/data/scoliosis/clean_new/Image_data/0715/'
    label1 = '/home/hjz/data/scoliosis/org/影像数据.xlsx'
    label2 = '/home/hjz/data/scoliosis/org/202407.xlsx'
    save_path = '/home/hjz/data/scoliosis/clean_new/Label/All'
    lab_maps = {}
    read_xlrx(label1,lab_maps, tabs=[1,0], cols = [[7,3],[3,4,5]], rename=os.listdir(path1))
    read_xlrx(label2,lab_maps, tabs=[0], cols = [[1,5,2,3,4]], rename = os.listdir(path2))
    # T = []
    # for key, lab_list in lab_maps.items():
    #     T.append(lab_list[4])
    # print(sorted(T))

    task_list = ['Cobb', 'Type', 'T', 'L', 'TL']
    for key in lab_maps.keys():
        lab_list = []
        for i, num in enumerate(lab_maps[key]):
            lab = split_cl(num, task_list[i])
            lab_list.append(lab)
        lab_maps[key] = lab_list
    
    train_data, test_data = split_train_test(lab_maps)
    print('train')
    Count_print(train_data, task_list)
    print('test')
    Count_print(test_data, task_list)
    print(len(lab_maps), len(train_data), len(test_data))

    # 写入
    with open(os.path.join(save_path,'train.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in train_data.keys():
            writer.writerow([key]+train_data[key])
    
    with open(os.path.join(save_path,'test.csv'), 'w') as file:
        writer = csv.writer(file)
        for key in test_data.keys():
            writer.writerow([key]+test_data[key])





Merge()

#main()
# data_clean()
