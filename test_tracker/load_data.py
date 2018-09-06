from PIL import Image
import numpy as np
import os
import tensorflow as tf
file_path = 'D:\CS\Computer Vision\cfnet_validation_set\cfnet-validation'
path = 'D:\CS\Computer Vision\cfnet_validation_set\cfnet-validation\\tc_Badminton_ce2\\'
image0 = tf.read_file(path+'0001.jpg')
image0 = tf.image.decode_jpeg(image0)
image0 = 255.0 * tf.image.convert_image_dtype(image0, tf.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

test = np.array(test)
print(np.shape(test))
print(image0)
print()
image = Image.open(path+'0001.jpg')
print(image.mode)
print(image.size)
image.show()

def get_dir(name):
    if name == '':
        return []
    name = name.replace('/', '\\')
    if name[-1] != '\\':
        name = name + '\\'
    file_list = os.listdir(name)
    dire = [x for x in file_list if os.path.isdir(name + x)]
    return dire

def get_ground_truth(name):
    f = open(name)
    feature = []
    for line in f.open(name):
        lines = line.strip().split("\t")
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return feature
    pass

dire = get_dir(file_path)
print(dire)
