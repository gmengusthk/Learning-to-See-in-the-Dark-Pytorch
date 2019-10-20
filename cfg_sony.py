import os

#data config
train_input_dir = '/home/gtmeng/guotaofiles/research/Learning-to-See-in-the-Dark/dataset/Sony/short/'
train_gt_dir = '/home/gtmeng/guotaofiles/research/Learning-to-See-in-the-Dark/dataset/Sony/long/'
train_crop_size=256


#train config
seed=0
train_batch_size=16
data_loader_num_workers=4
base_lr=1e-4
momentum=0.9
total_epochs=2000
lr_decay_epochs=[1000,1500]
lr_decay_rate=0.1
model_save_interval=100
model_save_path='./sony_snapshots'
log_interval=4

start_epoch=501

for epoch in lr_decay_epochs:
    if epoch<=start_epoch:
        base_lr*=lr_decay_rate

snapshot_path=os.path.join(model_save_path,'model_%05d.pth'%(start_epoch-1))
