import torch
import torch.optim as optim
import torch.nn as nn
from network import SeeInDarkNet
from raw_image_dataset import RawImageDatasetSony
import cfg_sony as cfg
import os

from torch.utils.tensorboard import SummaryWriter

def train(model, device, train_loader, loss_function, optimizer, epoch, tb_writer):
    model.train()

    batch_num=len(train_loader)
    sample_cnt=0.0
    loss_acc=0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc+=loss.item()
        sample_cnt+=1

        tb_writer.add_scalar('Loss/loss', loss, (epoch-1)*len(train_loader)+batch_idx)

        if (batch_idx+1) % cfg.log_interval == 0:
            loss_avg=loss_acc/sample_cnt
            loss_acc=0.0
            sample_cnt=0.0
            log_str='Train Epoch:%d %d/%d(%d%%) loss: %.6f smooth loss: %.6f'%(epoch,batch_idx+1,batch_num,(100*float(batch_idx+1)/batch_num),loss.item(),loss_avg)
            print(log_str)
    
    if sample_cnt>0:
        loss_avg=loss_acc/sample_cnt
        log_str='Train Epoch:%d %d/%d(%d%%) loss: %.6f smooth loss: %.6f'%(epoch,batch_idx+1,batch_num,(100*float(batch_idx+1)/batch_num),loss.item(),loss_avg)
        print(log_str)

def test():
    pass

if __name__=='__main__':

    use_cuda=torch.cuda.is_available()
    torch.manual_seed(cfg.seed)
    device=torch.device("cuda" if use_cuda else "cpu")
    print('device:',device)

    train_dataset=RawImageDatasetSony(cfg.input_dir,cfg.gt_dir,crop_size=cfg.train_crop_size)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=cfg.train_batch_size, shuffle=True,
                                            num_workers=cfg.data_loader_num_workers, pin_memory=True)


    model = SeeInDarkNet()
    model=model.to(device)

    if cfg.snapshot_path is not None:
        model.load_state_dict(torch.load(cfg.snapshot_path))

    loss_function=nn.L1Loss()
    loss_function=loss_function.to(device)

    # optimizer=optim.SGD(model.parameters(), lr=cfg.base_lr, momentum=cfg.momentum)
    optimizer=optim.Adam(model.parameters(), lr=cfg.base_lr)

    tb_writer = SummaryWriter()

    for epoch in range(cfg.start_epoch, cfg.total_epochs + 1):
        train(model, device, train_loader, loss_function, optimizer, epoch, tb_writer)

        if epoch in cfg.lr_decay_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr']=param_group['lr']*cfg.lr_decay_rate

        if epoch%cfg.model_save_interval==0:
            model_save_path=os.path.join(cfg.model_save_path,'model_%05d.pth'%(epoch))
            torch.save(model.state_dict(),model_save_path)