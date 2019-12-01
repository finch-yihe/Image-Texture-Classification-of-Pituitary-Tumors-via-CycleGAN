from glob import glob
import numpy as np
import scipy.misc

class DataLoader():
    def __init__(self):
        self.image_A = np.load(r'datas/T1_images_3072.npy') 
        self.image_B = np.load(r'datas/T2_images_1008.npy')

    def load_data(self, classification, batch_size=1):
        if classification == 0:
            imgs = self.image_A
            choose = self.image_A.shape[0]
        else:
            imgs = self.image_B
            choose = self.image_B.shape[0]
        batch_images = np.random.choice(choose, batch_size)

        resimgs = []
        for img_id in batch_images:
            img = imgs[img_id]
            if np.random.random() > 0.5:
                img = np.fliplr(img)
            resimgs.append(img)

        return np.array(resimgs)

    def load_batch(self, batch_size, is_testing=False):
        
        self.n_batches = int(min(self.image_A.shape[0], self.image_B.shape[0]) / batch_size)
        total_samples = self.n_batches * batch_size
        choose_A = np.random.choice(self.image_A.shape[0], total_samples, replace=False)
        choose_B = np.random.choice(self.image_B.shape[0], total_samples, replace=False)
        
        for i in range(self.n_batches-1):
            imgs_A, imgs_B = [], []
            imgs_A_id = choose_A[i*batch_size:(i+1)*batch_size]
            imgs_B_id = choose_B[i*batch_size:(i+1)*batch_size]

            for img_A_id, img_B_id in zip(imgs_A_id, imgs_B_id):
                
                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(self.image_A[img_A_id,:,:,:])
                    img_B = np.fliplr(self.image_B[img_B_id,:,:,:])
                else:
                    img_A = self.image_A[img_A_id,:,:,:]
                    img_B = self.image_B[img_B_id,:,:,:]

                imgs_A.append(img_A)
                imgs_B.append(img_B)
            yield np.array(imgs_A), np.array(imgs_B)


