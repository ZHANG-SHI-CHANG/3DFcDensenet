import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np
import cv2
import nrrd

from FCDenseNet import FCDenseNet as Model

from DataLoader import BatchGenerator

import os
import glob
import sys
import platform

if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

root = os.getcwd()

def PaddingData(datas,scale=4):
    row,col,cha = datas.shape
    if row%scale==0:
        pass
    else:
        padding = np.zeros((scale-(row-scale*(row//scale)),col,cha),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=0)
    row,col,cha = datas.shape
    if col%scale==0:
        pass
    else:
        padding = np.zeros((row,scale-(col-scale*(col//scale)),cha),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=1)
    row,col,cha = datas.shape
    if cha%scale==0:
        pass
    else:
        padding = np.zeros((row,col,scale-(cha-scale*(cha//scale))),dtype=datas.dtype)
        datas = np.concatenate((datas,padding),axis=2)
    return datas

if __name__=='__main__':
    input_size = int(sys.argv[1])
    
    data_path = sys.argv[2]
    data_name = data_path.split(SplitSym)[-1][:-5]
    label_path = data_path[:-5]+'_glm.nrrd'
    
    save_path = sys.argv[3]
    
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)
    
    model = Model(input_size=input_size)

    save_checkpoints_path = os.path.join(os.getcwd(),'checkpoints')
    latest_checkpoint = tf.train.latest_checkpoint(save_checkpoints_path)
    saver = tf.train.Saver()
    
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)
    print("Checkpoint loaded\n\n")
    
    data,option = nrrd.read(data_path)
    data = (data-data.min())*1.0/(data.max()-data.min())
    original_shape = data.shape
    data = PaddingData(data,input_size)
    
    label,_option = nrrd.read(label_path)
    label = PaddingData(label,input_size)

    padding_shape = list(map(lambda x:x[0]-x[1],zip(data.shape,original_shape)))
    
    result = np.zeros_like(data,dtype=np.float32)
    
    acc_list = []
    iou_list = []
    for ch in range(0,data.shape[2],input_size):
        for row in range(0,data.shape[0],input_size):
            for col in range(0,data.shape[1],input_size):
                print(row,col,ch)
                
                feed_data = data[row:row+input_size,col:col+input_size,ch:ch+input_size]
                label_data = label[row:row+input_size,col:col+input_size,ch:ch+input_size]
                
                feed_dict = {model.input_image:feed_data[np.newaxis,:,:,:,np.newaxis].astype(np.float32),
                             model.is_training:0.0,
                             model.dropout:1.0,
                             model.y:label_data[np.newaxis,:,:,:].astype(np.int32)}

                predict, acc, iou = sess.run(
                                             [model.y_out_argmax,model.accuracy,model.iou],
                                             feed_dict=feed_dict
                                            )
                
                acc_list.append(acc)
                iou_list.append(iou)
                print('ACC:{} IOU:{}'.format(acc,iou))
                
                result[row:row+input_size,col:col+input_size,ch:ch+input_size] = predict
                
                result[row:row+input_size,col:col+input_size,ch:ch+input_size] = feed_data

    #print('avg acc:{}, avg iou:{}'.format(np.mean(acc_list), np.mean(iou_list)))
    
    if padding_shape[0]>0:
        result = result[:-padding_shape[0],:,:]
    if padding_shape[1]>0:
        result = result[:,:-padding_shape[1],:]
    if padding_shape[2]>0:
        result = result[:,:,:-padding_shape[2]]
    
    print(original_shape)
    print(result.shape)
    
    nrrd.write(save_path+SplitSym+data_name+'_result.nrrd',result,options=option)