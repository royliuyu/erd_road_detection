'''
mini_imagenet: there are 60000 images, zip file of 3.1GB

mini_imagenet
 ├─labels.json
 ├─test.csv
 ├─val.csv
 ├─train.csv
 └─ images
     ├─ n0153282900000005.jpg
     ├─ n0153282900000006.jpg
     └─ ...

'''

import time
import torch
import argparse
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
root = os.environ['HOME']
import torchvision.models as models
import erd_cnn
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

parser =  argparse.ArgumentParser()
# parser.add_argument('--dataset', metavar = 'MiniImageNet', default= 'MiniImageNet', help =' to choose "MiniImageNet" or "imagenet" ')
parser.add_argument('--arch', metavar = 'arch', default = 'alexnet', help ='e.g. resnet50, alexnet')
parser.add_argument('--max_infer_num', metavar = 'max_infer_num', default = 200)

class MiniImageNet(Dataset):
    def __init__(self, mode, transform):
        self.data_dir = os.path.join(root,'./Documents/datasets/mini_imagenet')
        self.csv_name = mode + '.csv'
        self.transform = transform[mode]
        assert os.path.exists(self.data_dir), 'Root dir "%s" is not found.' % (self.data_dir)
        img_dir = os.path.join(self.data_dir, './images') # ./images
        assert os.path.exists(img_dir), 'Image dir "%s" is not found' % (img_dir)
        img_label_file = os.path.join(self.data_dir, self.csv_name)
        img_label = pd.read_csv(img_label_file)

        # generate dict for converting label (n01930112) to class_value (37)
        label_mapping = pd.read_csv(os.path.join(self.data_dir, '../LOC_synset_mapping.csv'), index_col=None)
        label_mapping_dict = {}
        for i in range(label_mapping.shape[0]):
            class_value, file_name, _,_ = label_mapping.loc[i]
            label_mapping_dict[file_name] = class_value

        self.img_path_list, self.label_list = [], []
        for i in range(img_label.shape[0]):
            self.img_path_list.append(os.path.join(img_dir, img_label.loc[i,'filename']))
            label_name = img_label.loc[i,'label'] # old label in format of n01930112
            self.label_list.append(label_mapping_dict[label_name])   # new label in class value format, e.g 37

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_path_list[idx])
        assert img.mode == 'RGB', 'image "%s" is not in RGB mode' % (self.img_path_list[idx])
        label = self.label_list[idx]
        if self.transform: img = self.transform(img)
        return img, label



def work(config):
    args = parser.parse_args()
    for key, value in config.items():  # update the args with transfered config
        vars(args)[key] = value

    # root = args.root
    model_name= args.arch
    batch_size = args.batch_size
    h,w = args.image_size
    device = args.device
    # dataset = args.dataset
    num_workers = args.workers

    transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((h,w)),  #224
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((h+32,w+32)), #256
                                   transforms.CenterCrop((h,w)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "test": transforms.Compose([transforms.Resize((h+32,w+32)),
                                    transforms.CenterCrop((h,w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    data_test = MiniImageNet('test', transform)
    dataload_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # assert dataset in ['MiniImageNet', 'imagenet'], f'Dataset \"{dataset}\" is not recognized!'

    if model_name == 'erd_cnn':
        model = erd_cnn.NET()
    else:
        model_func = 'models.'+ model_name
        # if dataset == 'MiniImageNet':
        #     data_test = MiniImageNet('test', transform)
        #     dataload_test = DataLoader(data_test, batch_size= batch_size, shuffle=True, num_workers=num_workers)
        # elif dataset == 'imagenet':
        #     dataload_test = imageNet.test_loader(batch_size=batch_size, workers=num_workers, image_size = (img_size,img_size))

        model = eval(model_func)(pretrained=True) # eval(): transform string to variable or function
    assert not ((not torch.cuda.is_available()) and (device =='cuda')), 'set device of cuda, while it is not available'

    model.to(device)
    model.eval()
    latency_total, latency, count = 0, 0, 0
    latency_list = []
    # quit_while = False
    print()
    print('Infer starts......... ', args)
    with torch.no_grad():

        # time.sleep(10)  # sleep 50 waiting for trainin
        for data, label in dataload_test:
            if device == 'cpu':
                start = time.time()
            else: #cuda
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()  #cuda
            data = data.to(device)
            prd = model.forward(data)
            prd = prd.to('cpu').detach()

            ## print result
            # result= prd.numpy()
            # result = np.argmax(result, axis=1)  # output is class_value
            # print(result)

            ## count latency
            if device == 'cpu':
                duration = (time.time() - start) *1000 # metrics in ms
            else:  # gpu
                ender.record()
                torch.cuda.synchronize()  ###
                duration = starter.elapsed_time(ender) # metrics in ms
            latency_list.append(duration)
            if count > args.max_infer_num: break
            count += 1
    latency = np.mean(latency_list[5:])  # take off GPU warmup
    print('Inferenc latency is: ', latency/batch_size, 'ms. Total ', count*batch_size,' images are infered.' )
    # print('acc1: ', acc1.numpy()[0]/100, ',        acc5: ', acc5.numpy()[0]/100)  # convert tensor to numpy


if __name__ == '__main__':
    model_list = ['mobilenet_v3_small', 'shufflenet_v2_x1_0', 'squeezenet1_1', 'efficientnet_b0','erd_cnn','efficientnet_v2_s','inception_v3']
    config ={'arch': 'inception_v3','workers': 1, 'batch_size': 1, 'image_size':(360, 640), 'device':'cuda', 'verbose': True}
    work(config)

