from torch.utils.data import Dataset
import glob, cv2, os
from PIL import Image

class myDataset(Dataset):
    def __init__(self, path, transform):
        self.data = np.load(path)
        self.train = self.data['train_images.npy'] #(1080, 28, 28, 3)
        if len(self.train.shape)==3:  #(1080, 28, 28)
            self.train = self.train[:,:,:,np.newaxis]
            self.train = np.repeat(self.train, 3, axis=3)
        self.transform = transform
        
    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, index):
        data = self.train[index,:,:,:]
        data = self.transform(Image.fromarray(data))
        return data, index
'''
dataset = myDataset(path='./data/organsmnist.npz',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

'''


class All_fundus_files(Dataset):
    def __init__(self, transform):
        self.transform = transform
        
        self.data_dir_list = [
            './data/fundus1/1_normal',
            './data/fundus1/2_cataract',
            './data/fundus1/2_glaucoma',
            './data/fundus1/3_retina_disease',
            './data/fundus2/test_images',
            './data/fundus2/train_images',
            './data/More/train'
        ]
        
        image_path = []
        for i,folder in enumerate(self.data_dir_list):
            ctlist=glob.glob(os.path.join(folder, '*.png'))
            image_path += ctlist
            #print(len(ctlist))
        print(len(image_path))
        self.image_path = image_path
        
        
    def __len__(self):
        return len(self.image_path)
            

    def __getitem__(self, index):
        fpath = self.image_path[index]
        input_data = cv2.imread(fpath) #numpy
        input_data = Image.fromarray(input_data[:,:,::-1]) #PIL.Image
        input_data = self.transform(input_data) 
        return input_data, index

'''
dataset = All_fundus_files(transform=transforms.Compose([
                           transforms.Resize(image_size),
                           transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
'''