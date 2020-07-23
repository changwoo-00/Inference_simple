import argparse
import os
import tensorflow as tf

tf.set_random_seed(20)
from model import infer

def get_params():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='dataset', help='path of the dataset')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')

    args = parser.parse_args()
    return args

def main(params):

    with tf.Session() as sess:
        model = infer(sess, params)
        model.test(params)
        
if __name__ == '__main__':
    params = vars(get_params())
    main(params)
