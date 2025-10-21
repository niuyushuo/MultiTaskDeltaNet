import numpy as np # linear algebra
import pandas as pd
import os

from PIL import Image
import re

from datasets.data_unet import UNETDataset


def load_dataset_s():

    def normalize(input_data):
        return (input_data.astype(np.float32))/255.0

    def denormalize(input_data):
        input_data = (input_data) * 255
        return input_data.astype(np.uint8)

    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'

    dirnames = os.listdir(path)
    #dirnames=sorted(dirnames, key=lambda s: float(re.findall(r'\d+', s)[0]))

    ###### Folder names and its corresponding image files
    file_names={}
    for dirname in dirnames:
        for root, dirnames, filenames_x in os.walk(path+'/'+dirname+'/results/img'):
            break
        filenames_x = sorted(filenames_x, key=lambda s: float(re.findall(r'\d+', s)[2]))
        if dirname == '201' or dirname == '203':
            filenames_x.pop(0)
        else:
            filenames_x.pop(0)
            filenames_x.pop(0)

        file_names[dirname]=filenames_x

    val_name1 = ['103']
    val_name2 = ['301']
    test_name1= ['201']
    test_name2= ['203']


    #### validation1:
    ##### Validation dataset:
    val_ref=[]
    val_res=[]

    val_label1=[]  ### validation label
    val_label2=[]

    for name in val_name1:    #### Loop all the folders

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            x_label1=np.array(x_label1)
            x_label2=np.array(x_label2)

            val_ref.append(x_ref)
            val_label1.append(x_label1[:,:,0])
            val_label2.append(x_label2[:,:,0])

    val_ref = np.array(val_ref)
    val_label1 = np.array(val_label1)
    val_label2 = np.array(val_label2)

    val_label1 = normalize(val_label1)
    val_label2 = normalize(val_label2)

    val_dataset1 = UNETDataset(val_ref,val_label1,val_label2,img_size=256,is_train=False,to_tensor=True)

    #### validation2:
    ##### Validation dataset:
    val_ref=[]
    val_res=[]

    val_label1=[]  ### validation label
    val_label2=[]

    for name in val_name2:    #### Loop all the folders

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            x_label1=np.array(x_label1)
            x_label2=np.array(x_label2)

            val_ref.append(x_ref)
            val_label1.append(x_label1[:,:,0])
            val_label2.append(x_label2[:,:,0])

    val_ref = np.array(val_ref)
    val_label1 = np.array(val_label1)
    val_label2 = np.array(val_label2)

    val_label1 = normalize(val_label1)
    val_label2 = normalize(val_label2)

    val_dataset2 = UNETDataset(val_ref,val_label1,val_label2,img_size=256,is_train=False,to_tensor=True)


    #### test1:
    ##### Test dataset:
    test_ref=[]
    test_res=[]

    test_label1=[]  ### validation label
    test_label2=[]
    test_map1=[]    ### visulization label
    test_map2=[]

    for name in test_name1:    #### Loop all the folders

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            x_label1=np.array(x_label1)
            x_label2=np.array(x_label2)

            test_ref.append(x_ref)
            test_label1.append(x_label1[:,:,0])
            test_label2.append(x_label2[:,:,0])

    test_ref = np.array(test_ref)
    test_label1 = np.array(test_label1)
    test_label2 = np.array(test_label2)

    test_label1 = normalize(test_label1)
    test_label2 = normalize(test_label2)

    test_dataset1 = UNETDataset(test_ref,test_label1,test_label2,img_size=256,is_train=False,to_tensor=True)


    #### test2:
    ##### Test dataset:
    test_ref=[]
    test_res=[]

    test_label1=[]  ### validation label
    test_label2=[]
    test_map1=[]    ### visulization label
    test_map2=[]

    for name in test_name2:    #### Loop all the folders

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            x_label1=np.array(x_label1)
            x_label2=np.array(x_label2)

            test_ref.append(x_ref)
            test_label1.append(x_label1[:,:,0])
            test_label2.append(x_label2[:,:,0])

    test_ref = np.array(test_ref)
    test_label1 = np.array(test_label1)
    test_label2 = np.array(test_label2)

    test_label1 = normalize(test_label1)
    test_label2 = normalize(test_label2)

    test_dataset2 = UNETDataset(test_ref,test_label1,test_label2,img_size=256,is_train=False,to_tensor=True)

    return val_dataset1, val_dataset2, test_dataset1, test_dataset2
