import numpy as np
import nrrd

import os
import glob
import sys
import platform

if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'

root = os.getcwd()

def match_data_label(path):
    datas = glob.glob( os.path.join(path,'datas','*.nrrd') )
    labels = []
    unneed = []
    for i,datas_path in enumerate(datas):
        datas_name = datas_path.split(SplitSym)[-1][:-5]
        if os.path.exists( os.path.join(path,'labels',datas_name+'_glm.nrrd') ):
            print('success match data:{} with label:{}'.format(datas_name+'.nrrd',datas_name+'_glm.nrrd'))
            labels.append(os.path.join(path,'labels',datas_name+'_glm.nrrd'))
        else:
            print('failure match datas:{}'.format(datas_name+'.nrrd'))
            unneed.append(i)
    if not unneed:
        for i in unneed[::-1]:
            del datas[i]
    
    dataset = list(zip(datas,labels))
    return dataset

if __name__=='__main__':
    try:
        input_size = int(sys.argv[1])
    except:
        input_size = 32
        
    stride = 16
    max_save = 1000
    save_threshold = 0.9
    
    if not os.path.exists(os.path.join(root,'processed_dataset','datas')):
        os.makedirs(os.path.join(root,'processed_dataset','datas'))
    if not os.path.exists(os.path.join(root,'processed_dataset','labels')):
        os.makedirs(os.path.join(root,'processed_dataset','labels')) 
    
    original_dataset = match_data_label(os.path.join(root,'original_dataset'))
    
    for i,(data_path,label_path) in enumerate(original_dataset):
        data_name = data_path.split(SplitSym)[-1][:-5]
        
        original_data,original_data_option = nrrd.read(data_path)
        original_data = (original_data-original_data.min())*1.0/(original_data.max()-original_data.min())
        original_label,original_label_option = nrrd.read(label_path)
        print('read {} complete max:{} min:{}'.format(data_name,original_data.max(),original_data.min()))
        
        save_count = 0
        
        for ch in range(0,original_data.shape[2],stride):
            for row in range(0,original_data.shape[0],stride):
                for col in range(0,original_data.shape[1],stride):
                    data = original_data[row:row+input_size,col:col+input_size,ch:ch+input_size]
                    label = original_label[row:row+input_size,col:col+input_size,ch:ch+input_size]
                    
                    if data.shape[0]==data.shape[1]==data.shape[2]==input_size: 
                        pass
                    else:
                        continue
                    
                    if np.sum((label>0).astype(np.float32)) >= int(input_size*input_size*input_size*save_threshold):
                        original_data_option['sizes'] = [data.shape[1],data.shape[0],data.shape[2]]
                        original_data_option['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
                        original_data_option['space origin'] = ['0','0','0']
                        original_label_option['sizes'] = [label.shape[1],label.shape[0],label.shape[2]]
                        original_label_option['space directions'] = [['1','0','0'],['0','1','0'],['0','0','1']]
                        original_label_option['space origin'] = ['0','0','0']
                        
                        if save_count<max_save:
                            print('ch:{} row:{} col:{}'.format(ch,row,col))
                            print('save data {} {},save label {} {}'.format(data_name+'.nrrd',save_count,data_name+'_glm.nrrd',save_count))
                            nrrd.write(os.path.join(root,'processed_dataset','datas',data_name+'_{}.nrrd'.format(save_count)),
                                       data,options=original_data_option)
                            nrrd.write(os.path.join(root,'processed_dataset','labels',data_name+'_{}_glm.nrrd'.format(save_count)),
                                       label,options=original_label_option)
                            save_count += 1
                        else:
                            continue