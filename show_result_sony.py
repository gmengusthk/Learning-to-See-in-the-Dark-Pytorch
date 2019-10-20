import torch
import torch.optim as optim
import torch.nn as nn
from network import SeeInDarkNet
from raw_image_dataset import RawImageDatasetSony,gt_2_cv
import cfg_sony as cfg
import os
import cv2

if __name__=='__main__':

    use_cuda=torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    print('device:',device)

    train_dataset=RawImageDatasetSony(cfg.train_input_dir,cfg.train_gt_dir,crop_size=1024)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=1, shuffle=False,
                                            num_workers=1, pin_memory=True)


    model = SeeInDarkNet()
    model=model.to(device)

    snapshot_path='./sony_snapshots/model_00500.pth'
    model.load_state_dict(torch.load(snapshot_path))

    image_write_dir='./viz'

    model.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        batch_size=target.size(0)
        for b_idx in range(batch_size):
            target_frame_cpu=target[b_idx].cpu().data
            output_frame_cpu=output[b_idx].cpu().data
            target_frame_cv=gt_2_cv(target_frame_cpu)
            output_frame_cv=gt_2_cv(output_frame_cpu)

            target_write_path=os.path.join(image_write_dir,'image_%03d_%2d_gt.png'%(batch_idx,b_idx))
            output_write_path=os.path.join(image_write_dir,'image_%03d_%2d_output.png'%(batch_idx,b_idx))
            cv2.imwrite(target_write_path,target_frame_cv)
            cv2.imwrite(output_write_path,output_frame_cv)
            print(target_write_path)
            print(output_write_path)
            print('-'*50)



   