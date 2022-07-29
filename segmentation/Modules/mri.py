##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 从三维取出2s张切片的代码
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import torch
import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from data_utils.utils import estimate_weights_mfb


# ---------------------数据载入--------------------------------------------
class LoadMRIData(Dataset):

    def __init__(self, mri_dir, list_dir, phase, num_class, num_slices=5, crop_size=32, se_loss=True, Encode3D=False,
                use_weight=False):
        "load MRI into a 2D slice and a 3D image"

        self.phase = phase
        self.se_loss = se_loss
        self.Encode3D = Encode3D
        self.num_class = num_class
        self.use_weight = use_weight
        self.num_slices = num_slices

        if self.use_weight:
            weight_dir = os.path.join(mri_dir, 'training-weightsnpy')
            self.weight_names = []
        # 数据载入
        if self.phase is 'train':
            data_dir = os.path.join(mri_dir, 'training-imagesnpy')
            if num_class is 6:
                label_dir = os.path.join(mri_dir, 'training-labels-remapnpy')
            else:
                label_dir = os.path.join(mri_dir, 'training-labels139')
            image_list = os.path.join(list_dir, 'train_pwml.txt')

            self.image_names = []
            self.image_slices = []
            self.label_names = []
            self.skull_names = []
            with open(image_list, 'r') as f:
                for line in f:  # 对每一个大脑进行处理，比如当前是01.nii数据，image_names:6400;skull
                    for i in range(256):  # i:0-255
                        image_name = os.path.join(data_dir,line.rstrip() + '.npy')  # ./image/img/train\1000_3.nii具体哪个文件，01.nii存了256次，每个数据存256次，那么25个，存6400次
                        label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')  # 6400个，每个存256次，一共6400
                        skull_name = os.path.join(data_dir, line.rstrip() + '_brainmask.npy')
                        self.image_names.append(image_name)
                        self.label_names.append(label_name)
                        self.skull_names.append(skull_name)
                        self.image_slices.append(i)  # image_slice是一维数组，存储0-255个数字，01.nii的256张切片

                        if self.use_weight:
                            weight_name = os.path.join(weight_dir, line.rstrip() + '_glm.npy')
                            self.weight_names.append(weight_name)
        elif self.phase is 'test':
            data_dir = os.path.join(mri_dir, 'testing-imagesnpy')
            if num_class is 6:
                label_dir = os.path.join(mri_dir, 'testing-labels-remapnpy')
            else:
                label_dir = os.path.join(mri_dir, 'testing-labels139')
            image_list = os.path.join(list_dir, 'text_pwml.txt')

            self.image_names = []
            self.image_slices = []
            self.label_names = []
            self.skull_names = []
            with open(image_list, 'r') as f:  # 对于测试集，仅仅存储了一次，一共15个，就各存一次
                for line in f:
                    image_name = os.path.join(data_dir, line.rstrip() + '.npy')
                    skull_name = os.path.join(data_dir, line.rstrip() + '_brainmask.npy')
                    label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                    self.image_names.append(image_name)
                    self.label_names.append(label_name)
                    self.skull_names.append(skull_name)

    def __getitem__(self, idx):  # 返回第idx样本的具体数据,idx=3623，对应15.nii的第39个切片
        # this is for non-pre-processing data
        image_name = self.image_names[idx]
        skull_name = self.skull_names[idx]# 第几idx个人的大脑图像数据，此处是nii三维数据，比如01.nii
        label_name = self.label_names[idx]  # 对应的第idx个人的label_names

        img_3D = np.load(image_name)
        skull_3D = np.load(skull_name) # 加载原图大脑图
        # normalize data
        img_3D = (img_3D.astype(np.float32) - 128) / 128  # 转换数据类型,将数据变为-1，类型试float32
        skull_3D = (skull_3D.astype(np.float32) - 128) / 128
        label_3D = np.load(label_name).astype(np.int32)  # 转换为int32类型256 256 256

        if self.phase is 'train':  # 先找到相对应的大脑图与标签，再根据train还是test找切片位置，然后二维堆叠的切片
            x_ax, y_ax, z_ax = np.shape(img_3D)

            image_slice = self.image_slices[idx]  # 对应的第idx张切片比如是39，第39张切片
            img_coronal = img_3D[:, :, image_slice][np.newaxis, :,:]  # np.newaxis的功能:插入新维度 1,256,256，np,newaxis的位置在哪里，就是在哪个方位增加了 一个维度
            img_c64 =img_coronal[0,96:160,96:160][np.newaxis, :,:]
            img_c128 =img_coronal[0,64:192,64:192][np.newaxis, :,:]
            label_coronal = label_3D[:, :, image_slice]  # 256 256
            skull_coronal = skull_3D[:, :, image_slice] # 1 256 256
            # img_coronal是属于slice那个切片的图像，lanel_coronal,skull_coronal同理,后续没有再用到img_coronal了
            sample = {'image': torch.from_numpy(img_coronal), 'label': torch.from_numpy(label_coronal),
                      'skull': torch.from_numpy(skull_coronal), 'img_64':torch.from_numpy(img_c64),'img_128':torch.from_numpy(img_c128)}

            # 对于不同的大脑图,curlabel是不同的值，shape也不同
            if self.se_loss:
                curlabel = np.unique(label_coronal)  # 该函数是去除数组中的重复数字，并进行排序之后输出。当前切片label的类别ndarray:(20,)[0 10 11 12 15 16 17 23 24 36]
                cls_logits = np.zeros(self.num_class, dtype=np.float32)  # (139,)[0 0 ]
                if np.sum(curlabel > self.num_class) > 0:  # 如果当前slice存在的标签值大于num_class
                    curlabel[curlabel > self.num_class] = 0
                cls_logits[curlabel] = 1  # cls_logits[0]=1#cls_logits:{ndarray:(139,)}因为curlabel没有超出139 num_class的值,所以将curlabel对应位置赋值为1
                sample['se_gt'] = torch.from_numpy(cls_logits)  # 将数组转换为张量

            if self.Encode3D:
                if image_slice <= int(self.num_slices * 2):  # image_slice<11
                    image_stack1 = img_3D[:, :, 0:int(image_slice)]  # 从0-4取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice+1):int(self.num_slices * 2 + 1)]# 从6-11取切片进行堆叠
                    skull_stack1 = skull_3D[:, :, 0:int(image_slice)]
                    skull_stack2 = skull_3D[:, :, int(image_slice+1):int(self.num_slices * 2 + 1)]
                elif image_slice == int(self.num_slices * 2+1):  # image_slice<11
                    image_stack1 = img_3D[:, :, 0:int(image_slice-1)]  # 从0-4取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice+1):int(self.num_slices * 2 + 1)]# 从6-11取切片进行堆叠
                    skull_stack1 = skull_3D[:, :, 0:int(image_slice-1)]
                    skull_stack2 = skull_3D[:, :, int(image_slice+1):int(self.num_slices * 2 + 1)]
                elif image_slice>=245 and image_slice < 255:  # image_slice>256-11=245
                    image_stack1 = img_3D[:, :, z_ax - int(self.num_slices * 2 + 1):int(image_slice)]  # 从256-11到256取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice+1):]
                    skull_stack1 = skull_3D[:, :, z_ax - int(self.num_slices * 2 + 1):int(image_slice)]
                    skull_stack2 = skull_3D[:, :, int(image_slice+1):]
                elif image_slice == 255:  # image_slice>256-11=245
                    image_stack1 = img_3D[:, :, z_ax - int(self.num_slices * 2 + 1):-1]  # 从256-11到256取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice+1):]
                    skull_stack1 = skull_3D[:, :, z_ax - int(self.num_slices * 2 + 1):-1]
                    skull_stack2 = skull_3D[:, :, int(image_slice+1):]
                else:
                    image_stack1 = img_3D[:, :,image_slice - self.num_slices:image_slice]
                    image_stack2 = img_3D[:, :, image_slice+1:image_slice + self.num_slices + 1]
                    skull_stack1 = skull_3D[:, :, image_slice - self.num_slices:image_slice]
                    skull_stack2 = skull_3D[:, :, image_slice+1:image_slice + self.num_slices + 1]
                image_stack1 = torch.from_numpy(image_stack1)
                image_stack2 = torch.from_numpy(image_stack2)
                skull_stack1 = torch.from_numpy(skull_stack1)
                skull_stack2 = torch.from_numpy(skull_stack2)
                image_stack=torch.cat((image_stack1,image_stack2),dim=2)
                skull_stack = torch.cat((skull_stack1, skull_stack2), dim=2)

                image_stack = np.transpose(image_stack, (2, 0, 1))
                skull_stack = np.transpose(skull_stack, (2, 0, 1))# 原本是（0，1，2）代表(x,y,z),(2,0,1)代表（z,x,y）
                sample['image_stack'] = image_stack  # 转换为张量，11 256 256
                sample['skull_stack'] = skull_stack

            # estimate class weights
            if self.use_weight:
                weight_name = self.weight_names[idx]
                weights_3D = np.load(weight_name).astype(np.float32)
                weight_slice = weights_3D[:, :, image_slice]
                sample['weights'] = torch.from_numpy(weight_slice)

        if self.phase is 'test':
            img_3D = np.transpose(img_3D, (2, 0, 1))  # 256 256 256（x,y,z转换为了在z,x,y）
            skull_3D = np.transpose(skull_3D, (2, 0, 1))
            label_3D = np.transpose(label_3D, (2, 0, 1))  # 256 256 256
            name = image_name.split('/')[-1][:-4]  # /为分割符，默认是所有空字符，name是1000_3这种格式的，[-1分割一次]，[-4]去掉.nii四位
            sample = {'image_3D': torch.from_numpy(img_3D), 'skull_3D': torch.from_numpy(skull_3D),'label_3D': torch.from_numpy(label_3D),
                      'name': name}

        return sample

    def __len__(self):
        return len(self.image_names)
