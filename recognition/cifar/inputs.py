import os
import re
import random
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image

top_dir = os.path.join(os.environ['HOME'], 'code', 'datasets','cifar-10-batches-py')
key_file=[
    'data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'
]
def unpickle(files):
    labels = []
    file_names = []
    datas = []
    for file in files:
        with open(file, 'rb') as fo:
            dic = pickle.load(fo, encoding='bytes')
            labels.extend(dic[b'labels'])
            file_names.extend(dic[b'filenames'])
            datas.extend(dic[b'data'])
    labels = np.array(labels)
    file_names = np.array(file_names)
    datas = np.array(datas)
    return labels, datas, file_names

def convert_image(datas, file_names, labels):
    zipped_data = [[file_names[i], np.reshape(datas[i], [3,32,32]).transpose(1,2,0), labels[i]] for i in range(len(datas))]
    return zipped_data

def show_images(count, zipped_data):
    for i in range(count):
        Image.fromarray(zipped_data[i][0]).show()

def get_data():
    files = [os.path.join(top_dir, file) for file in key_file]
    labels, datas, file_names = unpickle(files)
    zipped_data = convert_image(datas, file_names, labels)
    random.shuffle(zipped_data)
    num_train = int(len(zipped_data) * 0.8)
    train = zipped_data[:num_train]
    test = zipped_data[num_train:]
    return train, test

def all_batch(zipped_data):
    images = []
    labels = []
    for data in zipped_data:
        images.append(to_grayscale(data[1]))
        labels.append(data[2])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def batch(zipped_data):
    labels = []
    images = []
    for data in zipped_data:
        images.append(data[1])
        labels.append(data[2])
    images = np.array(images)
    labels = np.array(labels)
    queue = tf.train.slice_input_producer([images, labels])
    image = queue[0]
    label = queue[1]
    return tf.train.batch(
        [image, label],
        batch_size=32,
        capacity=len(zipped_data) * 2 * 3 * 32
    )

def to_grayscale(x):
    #print(np.dot(x[...,:3], [0.299, 0.587, 0.114]))
    return np.dot(x[...,:3], [0.299, 0.587, 0.114])

if __name__ == '__main__':
    files = [os.path.join(top_dir, file) for file in key_file]
    labels, datas, file_names = unpickle(files)

    print(datas.shape)

    #train, test = get_data()

    #images, labels = all_batch(train)
    #to_grayscale(images[0])
