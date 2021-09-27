import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import hydra

class JointLimDataSet(Dataset):
    def __init__(self, dataset_path, subject):

        dataset_path = hydra.utils.to_absolute_path(dataset_path)

        seqs = ['op2', 'op3', 'op4', 'op5', 'op7', 'op8', 'op9']

        shape = np.load(os.path.join(dataset_path, 'shapes', '%d_shape.npz'%subject))
        self.betas = torch.tensor(shape['betas'][:10])
        self.gender = str(shape['gender'])
        self.meta_info = {'betas': shape['betas'][:10], 'gender': self.gender}

        self.frame_names = []

        for seq in seqs:
            self.frame_names += sorted(glob.glob(os.path.join(dataset_path, 'points', str(subject), seq + '_poses'+'*.npz')))


    def __getitem__(self, index):

        name = self.frame_names[index]
        data = {}

        dataset = np.load(name)

        data['pts_d'] = torch.tensor(dataset['points'][0]).float()
        data['occ_gt'] = torch.tensor(dataset['occupancies'][0, :]).float().unsqueeze(-1)

        data['smpl_verts'] = torch.tensor(dataset['vertices'])
        data['smpl_tfs'] = torch.tensor(dataset['bone_transforms']).inverse()
        data['smpl_jnts'] = torch.tensor(dataset['joints'])
        data['smpl_thetas']  = torch.tensor(dataset['pose'])
        data['smpl_betas'] = self.betas

        return data
        
    def __len__(self):
        return len(self.frame_names)


class JointLimDataModule(pl.LightningDataModule):

    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):

        if stage == 'test':
            self.dataset_test = JointLimDataSet(dataset_path=self.opt.dataset_path,
                                                subject=self.opt.subject,
                                                )

            self.meta_info = self.dataset_test.meta_info

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset_test,
                                batch_size=1,
                                num_workers=self.opt.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader
