from PIL import Image
import os
file_path = 'D:\CS\Computer Vision\cfnet_validation_set\cfnet-validation'
path = 'D:\CS\Computer Vision\cfnet_validation_set\cfnet-validation\\tc_Badminton_ce2\\'
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

    pass

dire = get_dir(file_path)
print(dire)
