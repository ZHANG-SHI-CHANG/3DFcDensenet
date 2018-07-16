import nrrd
import numpy as np

import os
import glob
import sys
import platform

if platform.system()=='Windows':
    SplitSym = '\\'
else:
    SplitSym = '/'
    
root = os.getcwd()

class BatchGenerator():
    def __init__(self,datasetpath,input_size=32,batchsize=4,shuffle=True):
        self.DatasetPath = datasetpath
        self.InputSize = input_size
        self.BatchSize = batchsize
        self.Shuffle = shuffle
        
        self.__prepare()
        
        self.epoch_batch_count = 0
    
    def next(self):
        while True:
            input_data,input_label = self.__getitem__(self.epoch_batch_count)
            self.epoch_batch_count += 1
            if self.epoch_batch_count*self.BatchSize>self.__len__():
                self.epoch_batch_count = 0
                if self.Shuffle:
                    np.random.shuffle(self.dataset)
                print('------------------------------next epoch------------------------------')
            yield input_data,input_label
        
    def __getitem__(self,idx):
        l_round = idx*self.BatchSize
        r_round = (idx+1)*self.BatchSize
        
        if r_round>self.__len__():
            r_round = self.__len__()
            l_round = r_round - self.BatchSize
        
        input_data = np.zeros((self.BatchSize,self.InputSize,self.InputSize,self.InputSize,1),dtype=np.float32)
        input_label = np.zeros((self.BatchSize,self.InputSize,self.InputSize,self.InputSize),dtype=np.float32)
        
        count = 0
        
        for data_path,label_path in self.dataset[l_round:r_round]:
            data,data_option = nrrd.read(data_path)
            label,label_option = nrrd.read(label_path)
            
            input_data[count,:,:,:,0] = data.astype(np.float32)
            input_label[count,:,:,:] = label.astype(np.int32)
            
            count += 1
        
        return input_data,input_label
    
    def __prepare(self):
        datas = glob.glob( os.path.join(self.DatasetPath,'datas','*.nrrd') )
        labels = []
        unneed = []
        for i,datas_path in enumerate(datas):
            datas_name = datas_path.split(SplitSym)[-1][:-5]
            if os.path.exists( os.path.join(self.DatasetPath,'labels',datas_name+'_glm.nrrd') ):
                print('success match data:{} with label:{}'.format(datas_name+'.nrrd',datas_name+'_glm.nrrd'))
                labels.append(os.path.join(self.DatasetPath,'labels',datas_name+'_glm.nrrd'))
            else:
                print('failure match datas:{}'.format(datas_name+'.nrrd'))
                unneed.append(i)
        if not unneed:
            for i in unneed[::-1]:
                del datas[i]
        
        self.dataset = list(zip(datas,labels))
        if self.Shuffle:
            np.random.shuffle(self.dataset)
    
    def __len__(self):
        return len(self.dataset)

if __name__=='__main__':
    train_dataloader = BatchGenerator(os.path.join(root,'processed_dataset'))
    
    count = 0
    for input_data,input_label in train_dataloader.next():
        print(input_data.shape)
        print(np.unique(input_data))
        print(input_label.shape)
        print(np.unique(input_label))
        count += 1
        if count>10:
            break