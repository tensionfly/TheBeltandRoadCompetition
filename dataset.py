import os
import cv2
import numpy as np
import mxnet as mx

class DataSet(mx.gluon.data.Dataset):

    class_name =['Residential area','School','Industrial park','Railway station','Airport','Park',
                    'Shopping area', 'Administrative district', 'Hospital']

    def __init__(self,img_dir: str, dataset_index, txt_dir, transform=None,test=False, **kwargs):
        super(DataSet, self).__init__(**kwargs)

        if isinstance(dataset_index,str):
            with open(dataset_index) as f:
                self.dataset_index = [t.strip() for t in f.readlines()]
        elif isinstance(dataset_index,list):
            self.dataset_index=dataset_index
        else:
            raise ValueError('please give right type of dataset_index,list or str!')

        self.img_dir = img_dir
        self.txt_dir=txt_dir
        self.transform = transform

        self.test=test
    
    def __getitem__(self, idx):
        idx = self.dataset_index[idx]
        img_path = os.path.join(self.img_dir,idx+'.jpg')
        img = cv2.imread(img_path)
        h,w,c=img.shape
        img = cv2.resize(img,(128,128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        txt_feat=np.load(self.txt_dir+idx+'.npy')
        data=(img,txt_feat)
        if not self.test:
            label=int(idx.split('.')[0][-1])-1
        else:
            label=-1
        
        if self.transform is None:
            return data, label
        else:
            return self.transform(data, label)

    def __len__(self):
        return len(self.dataset_index)
