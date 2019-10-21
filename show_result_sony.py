import torch
import torch.optim as optim
import torch.nn as nn
from network import SeeInDarkNet
from raw_image_dataset import RawImageDatasetSony,gt_2_cv
import cfg_sony as cfg
import os
import cv2
import argparse

if __name__=='__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epoch', type=int, default=50,
                        help='input the epoch of the pretrained model (default: 50)')
    args = parser.parse_args()



    use_cuda=torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    print('device:',device)

    dataset=RawImageDatasetSony(cfg.input_dir,cfg.gt_dir,crop_size=1024,phase='test')

    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True)


    model = SeeInDarkNet()
    model=model.to(device)

    snapshot_path='./sony_snapshots/model_%05d.pth'%(args.epoch)
    print('test with %s'%(snapshot_path))
    model.load_state_dict(torch.load(snapshot_path))

    image_write_dir='./viz'

    model.eval()

    for batch_idx, (data, target, raw_image) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        batch_size=target.size(0)
        for b_idx in range(batch_size):
            target_frame_cpu=target[b_idx].cpu().data
            output_frame_cpu=output[b_idx].cpu().data
            raw_image_frame_cpu=raw_image[b_idx].cpu().data

            raw_image_frame_cpu=raw_image_frame_cpu/torch.mean(raw_image_frame_cpu)*torch.mean(target_frame_cpu)*0.9

            target_frame_cv=gt_2_cv(target_frame_cpu)
            output_frame_cv=gt_2_cv(output_frame_cpu)
            raw_image_frame_cv=gt_2_cv(raw_image_frame_cpu)


            target_write_path=os.path.join(image_write_dir,'image_%03d_%2d_gt.png'%(batch_idx,b_idx))
            output_write_path=os.path.join(image_write_dir,'image_%03d_%2d_output.png'%(batch_idx,b_idx))
            raw_image_write_path=os.path.join(image_write_dir,'image_%03d_%2d_raw.png'%(batch_idx,b_idx))
            cv2.imwrite(target_write_path,target_frame_cv)
            cv2.imwrite(output_write_path,output_frame_cv)
            cv2.imwrite(raw_image_write_path,raw_image_frame_cv)
            print(target_write_path)
            print(output_write_path)
            print(raw_image_write_path)
            print('-'*50)



   