import numpy as np
data0 = np.load(r'dataset/total_images_152.npy')
data1 = np.load(r'dataset/T1_total_images_152.npy')
data2 = np.load(r'dataset/T2_total_images_152.npy')
label = np.load(r'dataset/total_label_152.npy')

np.save(r'data/train/total_image_137.npy',data0[:137,:,:,:])
np.save(r'data/train/T1_total_image_137.npy',data1[:137,:,:,:])
np.save(r'data/train/T2_total_image_137.npy',data2[:137,:,:,:])
np.save(r'data/train/total_label_137.npy',label[:137])
np.save(r'data/test/total_image_15.npy',data0[137:,:,:,:])
np.save(r'data/test/T1_total_image_15.npy',data1[137:,:,:,:])
np.save(r'data/test/T2_total_image_15.npy',data2[137:,:,:,:])
np.save(r'data/test/total_label_15.npy',label[137:])