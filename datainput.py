import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
import cv2
import math

class DataLoader():
    def __init__(self, dataset_path, img_res=(128, 128)):
        self.dataset_path = dataset_path
        self.img_res = img_res

def chunkMaker(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
    return sum(l) / len(l)

def np_standardization(image):
    mean_ = np.mean(image)
    std_ = np.std(image)
    image = (image-mean_)/std_
    return image

def np_normalize(image):
    MIN_BOUND = np.min(image).astype(np.float32)
    MAX_BOUND = np.max(image).astype(np.float32)
    image = (image.astype(np.float32) - MIN_BOUND) / (MAX_BOUND - MIN_BOUND) * 2 - 1
    return image

def process_data(path, HM_SLICES=12, IMG_PX_SIZE=256, visualize=False):
        path = path
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        slices = [cv2.resize(np_normalize(each_slice.pixel_array),(int(IMG_PX_SIZE),int(IMG_PX_SIZE))) for each_slice in slices]
        if slices[0].shape != (256, 256):
            print(slices[0].shape)
        if len(slices) == HM_SLICES-1:
            slices.append(slices[-1])
        elif len(slices) == HM_SLICES-2:
            slices.insert(0, slices[0])
            slices.append(new_slices[-1])
        elif len(slices) == HM_SLICES+1:
            print('13')
            slices = slices[:12]
        elif len(slices) == HM_SLICES+2:
            print('14')
            slices = slices[1:13]
        elif len(slices) == HM_SLICES+3:
            print('15')
            slices = slices[1:13]
        elif len(slices) == HM_SLICES+4:
            print('16')
            slices = slices[2:14]

        if visualize:
            fig = plt.figure()
            for num,each_slice in enumerate(slices):
                y = fig.add_subplot(4,5,num+1)
                y.imshow(each_slice, cmap='gray')
            plt.show()
        return np.moveaxis(np.array(slices),0,2)

Initial_T1_images = []
Initial_T2_images = []
T1_label_withlabel = []
num = set()

path1 = r'/home/yihe/Documents/mission1/data/train/withlabel'
path2 = r'data/withoutlabel'

'''T1训练数据集'''
# for i in os.listdir(path1):
#     for j in os.listdir(os.path.join(path1, i)):
#         if "T1" in j and ('Ocor' in j or 'OCOR' in j):
#             if len(os.listdir(os.path.join(path1, i, j))) > 16:
#                 break
#             try:
#                 T1_image = process_data(os.path.join(path1, i, j))
#             except Exception as e:
#                 pass
#             finally:
#                 Initial_T1_images.append(T1_image)
# for i in os.listdir(path2):
#     for j in os.listdir(os.path.join(path2, i)):
#         if "T1" in j and ('Ocor' in j or 'OCOR' in j):
#             if len(os.listdir(os.path.join(path2, i, j))) > 16:
#                 break
#             try:
#                 T1_image = process_data(os.path.join(path2, i, j))
#             except Exception as e:
#                 pass
#             finally:
#                 Initial_T1_images.append(T1_image)
# images = np.array(Initial_T1_images).reshape(-1, 256, 256, 12)
# images = list(images)
# newdata = []
# for i in images:
#     newdata.append(np.array(np.fliplr(i)))
# images = np.array(images + newdata)
# np.save('data/T1_images_{}.npy'.format(images.shape[0]), images)



'''T2训练数据集'''
# for i in os.listdir(path1):
#     for j in os.listdir(os.path.join(path1, i)):
#         if "T2" in j and ('Ocor' in j or 'OCOR' in j):
#             if len(os.listdir(os.path.join(path1, i, j))) > 16:
#                 break
#             try:
#                 T1_image = process_data(os.path.join(path1, i, j))
#             except Exception as e:
#                 pass
#             finally:
#                 Initial_T1_images.append(T1_image)
# for i in os.listdir(path2):
#     for j in os.listdir(os.path.join(path2, i)):
#         if "T2" in j and ('Ocor' in j or 'OCOR' in j):
#             if len(os.listdir(os.path.join(path2, i, j))) > 16:
#                 break
#             try:
#                 T1_image = process_data(os.path.join(path2, i, j))
#             except Exception as e:
#                 pass
#             finally:
#                 Initial_T1_images.append(T1_image)
# images = np.array(Initial_T1_images).reshape(-1, 256, 256, 12)
# images = list(images)
# newdata = []
# for i in images:
#     newdata.append(np.array(np.fliplr(i)))
# images = np.array(images + newdata)
# np.save('data/T2_images_{}.npy'.format(images.shape[0]), images)









'''T1训练数据集带标签'''
# for i in os.listdir(path1):
#     for j in os.listdir(os.path.join(path1, i)):
#         if "T1" in j and ('Ocor' in j or 'OCOR' in j):
#             if len(os.listdir(os.path.join(path1, i, j))) > 16:
#                 break
#             try:
#                 T1_image = process_data(os.path.join(path1, i, j))
#             except Exception as e:
#                 pass
#             finally:
#                 Initial_T1_images.append(T1_image)
#                 T1_label_withlabel.append(int(i[-1])-1)
# images = np.array(Initial_T1_images)
# labels = np.array(T1_label_withlabel)
# n = int(images.shape[0])
# data = []
# label = []
# images = list(images)
# newdata = []
# for i in images:
#     newdata.append(np.array(np.fliplr(i)))
# for i in range(n):
#     data.append(images[i])
#     data.append(newdata[i])
# for i in range(n):
#     label.append(labels[i])
#     label.append(labels[i])
# data = np.array(data)
# label = np.array(label)
# np.save('dataset/T1_train_images_{}.npy'.format(data.shape[0]), data)
# np.save('dataset/T1_train_label_{}.npy'.format(label.shape[0]), label)






'''T2训练数据集带标签'''
# for i in os.listdir(path1):
#     for j in os.listdir(os.path.join(path1, i)):
#         if "T2" in j and ('Ocor' in j or 'OCOR' in j):
#             if len(os.listdir(os.path.join(path1, i, j))) > 16:
#                 break
#             try:
#                 T1_image = process_data(os.path.join(path1, i, j))
#             except Exception as e:
#                 pass
#             finally:
#                 Initial_T1_images.append(T1_image)
#                 T1_label_withlabel.append(int(i[-1])-1)
# images = np.array(Initial_T1_images)
# labels = np.array(T1_label_withlabel)
# n = int(images.shape[0])
# data = []
# label = []
# images = list(images)
# newdata = []
# for i in images:
#     newdata.append(np.array(np.fliplr(i)))
# for i in range(n):
#     data.append(images[i])
#     data.append(newdata[i])
# for i in range(n):
#     label.append(labels[i])
#     label.append(labels[i])
# data = np.array(data)
# label = np.array(label)
# np.save('dataset/T2_train_images_{}.npy'.format(data.shape[0]), data)
# np.save('dataset/T2_train_label_{}.npy'.format(label.shape[0]), label)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

translate1 = keras.models.load_model(r'model/g_AB_1006.model', custom_objects={"InstanceNormalization":InstanceNormalization})
translate2 = keras.models.load_model(r'model/g_BA_1006.model', custom_objects={"InstanceNormalization":InstanceNormalization})


T1data = list(np.load(r'dataset/T1_train_images_112.npy'))
T2data = list(np.load(r'dataset/T2_train_images_40.npy'))
T1label = list(np.load(r'dataset/T1_train_label_112.npy'))
T2label = list(np.load(r'dataset/T2_train_label_40.npy'))


new_data1 = []
for i in T1data:
    new_image = []
    for j in range(12):
        # new_image.append(i[:,:,j].reshape(1,256,256,1)[0,:,:,0])
        new_image.append(translate1.predict(i[:,:,j].reshape(1,256,256,1))[0,:,:,0])
    new_data1.append(np.array(new_image))
new_data1 = np.moveaxis(np.array(new_data1),1,3)
new_data2 = []
for i in T2data:
    new_image = []
    for j in range(12):
        # new_image.append(translate2.predict(i[:,:,j].reshape(1,256,256,1))[0,:,:,0])
        new_image.append(i[:,:,j].reshape(1,256,256,1)[0,:,:,0])
    new_data2.append(np.array(new_image))
new_data2 = np.moveaxis(np.array(new_data2),1,3)
new_data = np.array(list(new_data1)+list(new_data2))
np.save(r'dataset/T2_total_images_{}.npy'.format(new_data.shape[0]),new_data)
label = np.array(T1label+T2label)
np.save(r'dataset/T2_total_label_{}.npy'.format(label.shape[0]), label)


# for i in range(5):
#     data.append(data1[i*5])
#     del data1[i*5]
#     label.append(label1[i*5])
#     del label1[i*5]
# data = np.array(data)
# data1 = np.array(data1)
# label = np.array(label)
# label1 = np.array(label1)
# np.save('data/test/T2_test_images_{}.npy'.format(data.shape[0]), data)
# np.save('data/test/T2_test_label_{}.npy'.format(label.shape[0]), label)
# np.save('data/train/T2_train_images_{}.npy'.format(data1.shape[0]), data1)
# np.save('data/train/T2_train_label_{}.npy'.format(label1.shape[0]), label1)