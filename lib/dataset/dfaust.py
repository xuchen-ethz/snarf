import os
import glob
import yaml
import hydra
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class DFaustDataset(Dataset):
    def __init__(self, dataset_path, stage, subject, points_per_frame=2000):

        dataset_path = hydra.utils.to_absolute_path(dataset_path)
        split_path = hydra.utils.to_absolute_path('lib/dataset/dfaust_split.yml')
        with open(split_path, 'r') as stream:
            split = yaml.safe_load(stream)

        self.stage = stage
        
        shape = np.load(os.path.join(dataset_path, 'shapes', '%d_shape.npz'%subject))
        self.betas = torch.tensor(shape['betas'][:10])
        self.gender = str(shape['gender'])
        self.meta_info = {'betas': shape['betas'][:10], 'gender': self.gender}

        self.frame_names = []
        for name in split[str(subject)][stage]:
            frame_name_act = glob.glob(os.path.join(dataset_path, 'points', str(subject), str(subject)+ '_' + name+'_poses'+'*.npz'))
            self.frame_names.extend(frame_name_act)

        self.points_per_frame = points_per_frame
        self.total_points = 100000


    def __getitem__(self, i):
        name = self.frame_names[i]
        data = {}
        dataset = np.load(name)
        if self.stage == 'train':
            random_idx = torch.cat([torch.randint(0,self.total_points,[self.points_per_frame//8]), # 1//8 for bbox samples
                                   torch.randint(0,self.total_points,[self.points_per_frame])+self.total_points], # 1 for surface samples
                                  0)
            data['pts_d'] = torch.tensor(dataset['points'][0, random_idx]).float()
            data['occ_gt'] = torch.tensor(dataset['occupancies'][0, random_idx]).float().unsqueeze(-1)
        elif self.stage == 'val':
            data['pts_d'] = torch.tensor(dataset['points'][0, ::20]).float()
            data['occ_gt'] = torch.tensor(dataset['occupancies'][0, ::20]).float().unsqueeze(-1)
        elif self.stage == 'test':
            data['pts_d'] = torch.tensor(dataset['points'][0, :]).float()
            data['occ_gt'] = torch.tensor(dataset['occupancies'][0, :]).float().unsqueeze(-1)

        data['smpl_verts'] = torch.tensor(dataset['vertices'])
        data['smpl_tfs'] = torch.tensor(dataset['bone_transforms']).inverse()
        data['smpl_jnts'] = torch.tensor(dataset['joints'])
        data['smpl_thetas'] = torch.tensor(dataset['pose'])
        data['smpl_betas'] = self.betas

        return data

    def __len__(self):
        return len(self.frame_names)

class DFaustDataModule(pl.LightningDataModule):

    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):

        if stage == 'fit':
            self.dataset_train = DFaustDataset(dataset_path=self.opt.dataset_path,
                                                stage = 'train',
                                                subject=self.opt.subject,
                                                points_per_frame=self.opt.points_per_frame,
                                                )

            self.dataset_val   = DFaustDataset(dataset_path=self.opt.dataset_path,
                                            stage = 'val',
                                            subject=self.opt.subject,
                                            )

            self.meta_info = self.dataset_train.meta_info

        elif stage == 'test':
            self.dataset_test   = DFaustDataset(dataset_path=self.opt.dataset_path,
                                                stage = 'test',
                                                subject=self.opt.subject,
                                                )

            self.meta_info = self.dataset_test.meta_info

    def train_dataloader(self):
        dataloader = DataLoader(self.dataset_train,
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.dataset_val,
                                batch_size=1,
                                num_workers=self.opt.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset_test,
                                batch_size=1,
                                num_workers=self.opt.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader
