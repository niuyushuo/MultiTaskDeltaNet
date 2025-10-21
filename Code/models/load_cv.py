import numpy as np # linear algebra
import pandas as pd
import os

from PIL import Image
import re

from datasets.VER_dataset import VERDataset, CARDataset


def load_dataset():

    def normalize(input_data):
        return (input_data.astype(np.float32))/255.0

    def denormalize(input_data):
        input_data = (input_data) * 255
        return input_data.astype(np.uint8)


    palette3 = np.array([[1, 1, 1],  # "no change"
                    [0.5, 0, 0],  # "appearing(1)"
                    [0, 0.5, 0],  # "disappearing(-1)"
                    [0.4,0.4,0.5]], dtype='float32')  # "overlap(2)"  [0.4,0.4,0.5]


    path='/home/yun13001/dataset/Carbon/tianyu_new_data/New_distribution/'

    dirnames = os.listdir(path)
    dirnames=sorted(dirnames, key=lambda s: float(re.findall(r'\d+', s)[0]))

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

    
    train_name=['102_R1','102_R2', '302']
    val_name = ['103','301']
    test_name= ['201','203']
    

    #### train:
    ##### Training dataset:
    train_ref=[]
    train_res=[]

    train_label1=[]  ### training label
    train_label2=[]
    train_map1=[]    ### visulization label
    train_map2=[] 
    
    for name in train_name:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]
        l = []

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)
     

        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)

        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                train_ref.append(ref[m])
                train_res.append(ref[n])

                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                train_label1.append(img_tt)
                train_map1.append(mask)
                ######  ######
            

                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                train_label2.append(img_tt)
                train_map2.append(mask)
                ######  ######

    train_ref = np.array(train_ref)
    train_res = np.array(train_res) 

    train_ref = denormalize(train_ref)
    train_res = denormalize(train_res)

    train_label1 = np.array(train_label1)
    train_label2 = np.array(train_label2)
    train_map1 = np.array(train_map1)
    train_map2 = np.array(train_map2)

    
    #### validation:
    ##### Validation dataset:
    val_ref=[]
    val_res=[]

    val_label1=[]  ### validation label
    val_label2=[]
    val_map1=[]    ### visulization label
    val_map2=[]

    for name in val_name:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)
        

        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                val_ref.append(ref[m])
                val_res.append(ref[n])

                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                val_label1.append(img_tt)
                val_map1.append(mask)
                ######  ######


                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                val_label2.append(img_tt)
                val_map2.append(mask)
                ######  ######

    val_ref = np.array(val_ref)
    val_res = np.array(val_res)

    val_ref = denormalize(val_ref)
    val_res = denormalize(val_res)

    val_label1 = np.array(val_label1)
    val_label2 = np.array(val_label2)
    val_map1 = np.array(val_map1)
    val_map2 = np.array(val_map2)


    #### test:
    ##### Test dataset:
    test_ref=[]
    test_res=[]

    test_label1=[]  ### validation label
    test_label2=[]
    test_map1=[]    ### visulization label
    test_map2=[]

    for name in test_name:    #### Loop all the folders

        ref=[]
        label1=[]
        label2=[]

        imgs = file_names[name]
        for i in range(len(imgs)):
            x_ref = Image.open(path+name+'/results/img/'+imgs[i]).convert('RGB')
            x_label1 = Image.open(path+name+'/results/area1/'+imgs[i]).convert('RGB')
            x_label2 = Image.open(path+name+'/results/area2/'+imgs[i]).convert('RGB')

            ref.append(x_ref)
            label1.append(x_label1)
            label2.append(x_label2)

            x_ref = np.array(x_ref)
            x_label1 = np.array(x_label1)
            x_label2 = np.array(x_label2)


        ref = np.array(ref)
        label1 = np.array(label1)
        label2 = np.array(label2)

        ref = normalize(ref)
        label1=normalize(label1)
        label2=normalize(label2)


        for m in range(len(ref)):

            numb =m

            #for n in range(numb):
            for n in range(len(ref)):

                test_ref.append(ref[m])
                test_res.append(ref[n])

                ###### Label1 ######
                x_ref = label1[m]
                x_res = label1[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                test_label1.append(img_tt)
                test_map1.append(mask)
                ######  ######


                ###### Label2 ######
                x_ref = label2[m]
                x_res = label2[n]

                #### overlap
                img = x_ref + x_res
                img_test=img[:,:,0]
                img_ol=img_test.copy()
                img_ol[(img_test==0)]=4.0
                ####

                #### difference
                img = x_ref - x_res
                img_test=img[:,:,0]
                img_tt=img_test.copy()
                img_tt[(img_test<0)]=2
                img_tt[(img_ol==4.0)]=3

                img_tt=np.int32(img_tt)
                mask = palette3[img_tt.ravel()].reshape(img.shape)
                ####

                test_label2.append(img_tt)
                test_map2.append(mask)
                ######  ######

    test_ref = np.array(test_ref)
    test_res = np.array(test_res)

    test_ref = denormalize(test_ref)
    test_res = denormalize(test_res)

    test_label1 = np.array(test_label1)
    test_label2 = np.array(test_label2)
    test_map1 = np.array(test_map1)
    test_map2 = np.array(test_map2)
    
    ##### only for cv
    train_dataset = CARDataset(train_ref,train_res,train_label1,train_label2,img_size=256,is_train=True,to_tensor=True)
    val_dataset = CARDataset(val_ref,val_res,val_label1,val_label2,img_size=256,is_train=False,to_tensor=True)
    test_dataset = CARDataset(test_ref,test_res,test_label1,test_label2,img_size=256,is_train=False,to_tensor=True)


    return train_dataset, val_dataset, test_dataset 
