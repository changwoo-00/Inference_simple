from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
import cv2
from utils import *
import utils

import cv2

class infer(object):
    def __init__(self, sess, params):
        self.sess = sess
        self.batch_size = params['batch_size']
        self.dataset_dir = params['dataset_dir']
        self.checkpoint_dir = './checkpoint/'
        self.test_dir =  './test/'

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
                
    def load_batch(self, input_path_list, idx):
        self.input_batch_list = []
        for i in range(self.batch_size):
            temp_img = cv2.imread(input_path_list[idx*self.batch_size + i], 0)
            w, h = temp_img.shape
            temp_img = np.expand_dims(temp_img[int(h/2)-128:int(h/2)+128,int(w/2)-128:int(w/2)+128],2)
            self.input_batch_list.append(temp_img)
        

    def load_data(self, dir_path):
        self.input_path_list = glob(dir_path+'/*')
        
    def test(self, params):
        meta_path = glob(self.checkpoint_dir+'*.meta')
        saver = tf.train.import_meta_graph(meta_path[0])
        saver.restore(self.sess, meta_path[0][:-5])

        input = self.sess.graph.get_tensor_by_name('test_B:0')
        input_1 = self.sess.graph.get_tensor_by_name('real_A_and_B_images:0')
        output = self.sess.graph.get_tensor_by_name('generatorB2A_3/Tanh:0')

        self.load_data(self.dataset_dir)

        batch_idxs = int(len(self.input_path_list)/self.batch_size)

        for idx in range(0, batch_idxs):

            self.load_batch(self.input_path_list, idx)
            t = np.concatenate((self.input_batch_list,self.input_batch_list),axis=3)
            
            img_out = self.sess.run([output], feed_dict={input:self.input_batch_list})
            #img_out = self.sess.run([output], feed_dict={input_1:t})

            for i in range(len(img_out)):
                temp_img = np.squeeze((img_out[i]+1)*128)
                cv2.imwrite(self.test_dir+str(idx)+'_'+str(i)+'.bmp',temp_img)