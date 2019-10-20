import torch
import torch.utils.data as torch_data
import glob
import os
import numpy as np
import rawpy
import cv2


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=0)
    img_shape = im.shape
    H = img_shape[1]
    W = img_shape[2]

    out = np.concatenate((im[:,0:H:2, 0:W:2],
                          im[:,0:H:2, 1:W:2],
                          im[:,1:H:2, 1:W:2],
                          im[:,1:H:2, 0:W:2]), axis=0)
    return out


def input_2_cv(input_tensor):
    input_tensor_numpy=input_tensor.numpy()
    input_tensor_numpy_list=[]
    for tensor in input_tensor_numpy:
        tensor_cv=np.clip(tensor*255.0,0,255)
        tensor_cv=np.uint8(tensor_cv)
        input_tensor_numpy_list.append(tensor_cv)
    return input_tensor_numpy_list
    

def gt_2_cv(gt_tensor):
    gt_tensor_numpy=gt_tensor.numpy()
    gt_tensor_numpy=gt_tensor_numpy.transpose(1,2,0)
    gt_tensor_numpy=np.clip(gt_tensor_numpy*255.0,0,255)
    gt_tensor_numpy=np.uint8(gt_tensor_numpy)
    return gt_tensor_numpy



class RawImageDatasetSony_Memory(torch_data.Dataset):
    def __init__(self,input_dir,gt_dir,crop_size=256,phase='train'):
        super(RawImageDatasetSony_Memory).__init__()

        self.input_dir=input_dir
        self.gt_dir=gt_dir
        # if crop_size=-1, output the image in original size 
        self.crop_size=crop_size

        if phase is 'train':
            fns = glob.glob(os.path.join(gt_dir,'0*.ARW'))
        else:
            fns = glob.glob(os.path.join(gt_dir,'1*.ARW'))
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in fns]

        self.in_files_list=[]
        self.gt_files_list=[]

        for file_id in self.ids:
            self.in_files_list.append(glob.glob(os.path.join(self.input_dir,'%05d_00*.ARW' % file_id)))
            self.gt_files_list.append(glob.glob(os.path.join(gt_dir,'%05d_00*.ARW' % file_id))[0])


        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * len(self.ids)
        self.input_images = {}
        self.input_images[300] = [None] * len(self.ids)
        self.input_images[250] = [None] * len(self.ids)
        self.input_images[100] = [None] * len(self.ids)

    def __getitem__(self,index):
        in_files=self.in_files_list[index]
        in_path=np.random.choice(in_files)
        in_exposure=float(os.path.basename(in_path)[9:-5])

        gt_path=self.gt_files_list[index]
        gt_exposure=float(os.path.basename(gt_path)[9:-5])
        exposure_ratio=min(gt_exposure/in_exposure,300.0)


        if self.input_images[int(exposure_ratio)][index] is None:
            in_raw=rawpy.imread(in_path)
            self.input_images[int(exposure_ratio)][index]=pack_raw(in_raw)*exposure_ratio

        if self.gt_images[index] is None:
            gt_raw=rawpy.imread(gt_path)
            gt_im=gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_im=gt_im.transpose(2,0,1)
            self.gt_images[index] =np.float32(gt_im / 65535.0)
        
        input_full_size_image=self.input_images[int(exposure_ratio)][index]
        gt_full_size_image=self.gt_images[index]
        
        # crop
        if self.crop_size>0:
            H,W=input_full_size_image.shape[1:3]

            xx=np.random.randint(0, W-self.crop_size)
            yy=np.random.randint(0, H-self.crop_size)

            input_patch=input_full_size_image[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
            gt_patch=gt_full_size_image[:, 2*yy:2*yy+2*self.crop_size, 2*xx:2*xx+2*self.crop_size]
        else:
            input_patch=input_full_size_image
            gt_patch=gt_full_size_image


        input_patch = np.minimum(input_patch, 1.0)

        # random flip and transpose
        if np.random.randint(2)==1:
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2)==1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2)==1:
            input_patch = np.transpose(input_patch, (0, 2, 1))
            gt_patch = np.transpose(gt_patch, (0, 2, 1))
        
        input_patch=np.ascontiguousarray(input_patch)
        gt_patch=np.ascontiguousarray(gt_patch)
        input_patch_torch=torch.from_numpy(input_patch)
        gt_patch_torch=torch.from_numpy(gt_patch)

        return input_patch_torch,gt_patch_torch
    
    def __len__(self):
        return len(self.ids)


class RawImageDatasetSony(torch_data.Dataset):
    def __init__(self,input_dir,gt_dir,crop_size=256,phase='train'):
        super(RawImageDatasetSony).__init__()

        self.input_dir=input_dir
        self.gt_dir=gt_dir
        # if crop_size=-1, output the image in original size
        self.crop_size=crop_size

        if phase is 'train':
            fns = glob.glob(os.path.join(gt_dir,'0*.ARW'))
        else:
            fns = glob.glob(os.path.join(gt_dir,'1*.ARW'))
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in fns]

        self.in_files_list=[]
        self.gt_files_list=[]

        for file_id in self.ids:
            self.in_files_list.append(glob.glob(os.path.join(self.input_dir,'%05d_00*.ARW' % file_id)))
            self.gt_files_list.append(glob.glob(os.path.join(gt_dir,'%05d_00*.ARW' % file_id))[0])

    def __getitem__(self,index):
        in_files=self.in_files_list[index]
        in_path=np.random.choice(in_files)
        in_exposure=float(os.path.basename(in_path)[9:-5])

        gt_path=self.gt_files_list[index]
        gt_exposure=float(os.path.basename(gt_path)[9:-5])
        exposure_ratio=min(gt_exposure/in_exposure,300.0)

        in_raw=rawpy.imread(in_path)
        input_full_size_image=pack_raw(in_raw)*exposure_ratio

        gt_raw=rawpy.imread(gt_path)
        gt_im=gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_im=gt_im.transpose(2,0,1)
        gt_full_size_image=np.float32(gt_im / 65535.0)
        
        
        # crop
        if self.crop_size>0:
            H,W=input_full_size_image.shape[1:3]

            xx=np.random.randint(0, W-self.crop_size)
            yy=np.random.randint(0, H-self.crop_size)

            input_patch=input_full_size_image[:, yy:yy + self.crop_size, xx:xx + self.crop_size]
            gt_patch=gt_full_size_image[:, 2*yy:2*yy+2*self.crop_size, 2*xx:2*xx+2*self.crop_size]
        else:
            input_patch=input_full_size_image
            gt_patch=gt_full_size_image

        input_patch = np.minimum(input_patch, 1.0)

        # random flip and transpose
        if np.random.randint(2)==1:
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2)==1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2)==1:
            input_patch = np.transpose(input_patch, (0, 2, 1))
            gt_patch = np.transpose(gt_patch, (0, 2, 1))
        
        input_patch=np.ascontiguousarray(input_patch)
        gt_patch=np.ascontiguousarray(gt_patch)
        input_patch_torch=torch.from_numpy(input_patch)
        gt_patch_torch=torch.from_numpy(gt_patch)

        return input_patch_torch,gt_patch_torch
    
    def __len__(self):
        return len(self.ids)




if __name__=='__main__':
    
    viz_dir='./viz'

    input_dir = '/home/gtmeng/guotaofiles/research/Learning-to-See-in-the-Dark/dataset/Sony/short/'
    gt_dir = '/home/gtmeng/guotaofiles/research/Learning-to-See-in-the-Dark/dataset/Sony/long/'

    dataset=RawImageDatasetSony_Memory(input_dir,gt_dir)
    
    for idx in range(len(dataset)):
        input_patch,gt_patch=dataset[idx]
        
        gt_patch_cv=gt_2_cv(gt_patch)
        gt_save_path=os.path.join(viz_dir,'img_%04d_gt.png'%(idx))
        cv2.imwrite(gt_save_path,gt_patch_cv)
        print(gt_save_path)

        input_patch_cv_list=input_2_cv(input_patch)
        for channel_idx in range(len(input_patch_cv_list)):
            input_save_path=os.path.join(viz_dir,'img_%04d_input_%d.png'%(idx,channel_idx))
            cv2.imwrite(input_save_path,input_patch_cv_list[channel_idx])
            print(input_save_path)

        
    

        
        

