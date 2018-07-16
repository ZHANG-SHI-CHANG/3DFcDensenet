import tensorflow as tf
from tensorflow.python.framework import graph_util

import numpy as np
import cv2

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

max_to_keep = 4
save_model_every = 1
test_every = 1

is_train = True

def main(input_size=32,batch_size=1,epoch=10000):
    tf.reset_default_graph()

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    trainer = Train(sess,input_size=input_size,batch_size=batch_size,epoch=epoch)

    if is_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n\n")
        except KeyboardInterrupt:
            trainer.save_model()
    else:
        print("Testing...")
        trainer.test()
        print("Testing Finished\n\n")

class Train:
    def __init__(self, sess, input_size=32, batch_size=1, epoch=10000):
        self.sess = sess
        self.input_size = input_size
        self.batch_size = batch_size
        self.epoch = epoch

        self.train_data = BatchGenerator(
                                         datasetpath=os.path.join(root,'processed_dataset'),input_size=self.input_size,batchsize=self.batch_size
                                        )
        
        print("Building the model...")
        self.model = Model(num_classes=22,input_size=self.input_size)
        print("Model is built successfully\n\n")
        
        self.saver = tf.train.Saver(max_to_keep=max_to_keep,
                                    keep_checkpoint_every_n_hours=10)
        
        self.save_checkpoints_path = os.path.join(os.getcwd(),'checkpoints')
        if not os.path.exists(self.save_checkpoints_path):
            os.mkdir(self.save_checkpoints_path)

        self.init = None
        self.__init_model()

        self.__load_model()
        
        summary_dir = os.path.join(os.getcwd(),'logs')
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)
        summary_dir_train = os.path.join(summary_dir,'train')
        if not os.path.exists(summary_dir_train):
            os.mkdir(summary_dir_train)
        summary_dir_test = os.path.join(summary_dir,'test')
        if not os.path.exists(summary_dir_test):
            os.mkdir(summary_dir_test)
        self.train_writer = tf.summary.FileWriter(summary_dir_train,sess.graph)
        self.test_writer = tf.summary.FileWriter(summary_dir_test)

    def __init_model(self):
        print("Initializing the model...")
        self.init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
        self.sess.run(self.init)
        print("Model initialized\n\n")

    def save_model(self):
        print("Saving a checkpoint")
        self.saver.save(self.sess, self.save_checkpoints_path+SplitSym+'model', self.model.global_epoch_tensor)
        print("Checkpoint Saved\n\n")
        
    def __load_model(self):
        latest_checkpoint = tf.train.latest_checkpoint(self.save_checkpoints_path)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
            print("Checkpoint loaded\n\n")
        else:
            print("First time to train!\n\n")
    
    def train(self):
        for cur_epoch in range(self.model.global_epoch_tensor.eval(self.sess) + 1, self.epoch + 1, 1):

            batch = 0
            
            loss_list = []
            acc_list = []
            iou_list = []
            
            for input_data,input_label in self.train_data.next():
                print('Training epoch:{},batch:{}'.format(cur_epoch,batch))
                
                cur_step = self.model.global_step_tensor.eval(self.sess)
                
                feed_dict={self.model.input_image:input_data,
                           self.model.is_training:1.0,
                           self.model.dropout:0.2,
                           self.model.y:input_label}
                
                _, _, loss, acc, iou, summaries_merged = self.sess.run(
                    [self.model.train_op, self.model.update_op ,self.model.all_loss, self.model.accuracy, self.model.iou, self.model.summaries_merged],
                    feed_dict=feed_dict)
                    
                print('batch-'+str(batch)+'|'+'loss:'+str(loss)+'|'+'acc:'+str(acc)+'|'+'iou:'+str(iou)+'\n')

                loss_list += [loss]
                acc_list += [acc]
                iou_list += [iou]

                self.model.global_step_assign_op.eval(session=self.sess,
                                                      feed_dict={self.model.global_step_input: cur_step + 1})

                self.train_writer.add_summary(summaries_merged,cur_step)

                if batch*self.batch_size > self.train_data.__len__():
                    batch = 0
                
                    avg_loss = np.mean(loss_list).astype(np.float32)
                    avg_acc = np.mean(acc_list).astype(np.float32)
                    avg_iou = np.mean(iou_list).astype(np.float32)
                    
                    self.model.global_epoch_assign_op.eval(session=self.sess,
                                                           feed_dict={self.model.global_epoch_input: cur_epoch + 1})

                    print('Epoch-'+str(cur_epoch)+'|'+'avg loss:'+str(avg_loss)+'|'+'avg acc:'+str(avg_acc)+'|'+'avg iou:'+str(avg_iou)+'\n')
                    break
                
                if batch==0 and cur_epoch%99==0:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    
                    _,summaries_merged = self.sess.run([self.model.train_op, self.model.summaries_merged],
                                                   feed_dict=feed_dict,
                                                   options=run_options,
                                                   run_metadata=run_metadata)
                    
                    self.train_writer.add_run_metadata(run_metadata, 'epoch{}batch{}'.format(cur_epoch,cur_step))
                    self.train_writer.add_summary(summaries_merged, cur_step)

                batch += 1
            
            if cur_epoch % save_model_every == 0 and cur_epoch != 0:
                self.save_model()
            
            if cur_epoch % test_every == 0:
                print('start test')
                self.test()
                print('end test')
    def test(self):
        pass

if __name__=='__main__':
    input_size = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    epoch = int(sys.argv[3])
    
    main(input_size,batch_size,epoch)
