import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import hydra

import pickle
import kaolin


class CAPEDataSet(Dataset):

    def __init__(self, dataset_path, subject=32, clothing='longshort'):

        dataset_path = hydra.utils.to_absolute_path(dataset_path)

        self.regstr_list = glob.glob(os.path.join(dataset_path, 'cape_release', 'sequences', '%05d'%subject, clothing+'_**/*.npz'), recursive=True)

        genders_list =  os.path.join(dataset_path, 'cape_release', 'misc', 'subj_genders.pkl') 
        with open(genders_list,'rb') as f:
            self.gender = pickle.load(f, encoding='latin1')['%05d'%subject]

        minimal_body_path = os.path.join(dataset_path, 'cape_release', 'minimal_body_shape', '%05d'%subject, '%05d_minimal.npy'%subject)
        self.v_template = np.load(minimal_body_path)

        self.meta_info = {'v_template': self.v_template, 'gender': self.gender}


    def __getitem__(self, index):

        data = {}

        while True:
            try:
                regstr = np.load(self.regstr_list[index])
                poses = regstr['pose']
                break
            except:
                index = np.random.randint(self.__len__())
                print('corrupted npz')

        verts = regstr['v_posed'] - regstr['transl'][None,:]
        verts = torch.tensor(verts).float()

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = 1
        smpl_params[4:76] = torch.tensor(poses).float()

        data['scan_verts'] = verts
        data['smpl_params'] = smpl_params
        data['smpl_thetas'] = smpl_params[4:76]
        data['smpl_betas'] = smpl_params[76:]

        return data

    def __len__(self):
        return len(self.regstr_list)

''' Used to generate groud-truth occupancy and bone transformations in batchs during training '''
class CAPEDataProcessor():

    def __init__(self, opt, meta_info, **kwargs):
        from lib.model.smpl import SMPLServer

        self.opt = opt
        self.gender = meta_info['gender']
        self.v_template =meta_info['v_template']

        self.smpl_server = SMPLServer(gender=self.gender, v_template=self.v_template)
        self.smpl_faces = torch.tensor(self.smpl_server.smpl.faces.astype('int')).unsqueeze(0).cuda()
        self.sampler = hydra.utils.instantiate(opt.sampler)

    def process(self, data):

        smpl_output = self.smpl_server(data['smpl_params'], absolute=True)
        data.update(smpl_output)

        num_batch, num_verts, num_dim = smpl_output['smpl_verts'].shape

        random_idx = torch.randint(0, num_verts, [num_batch, self.opt.points_per_frame,1], device=smpl_output['smpl_verts'].device)
        
        random_pts = torch.gather(data['scan_verts'], 1, random_idx.expand(-1, -1, num_dim))
        data['pts_d']  = self.sampler.get_points(random_pts)

        data['occ_gt'] = kaolin.ops.mesh.check_sign(data['scan_verts'], self.smpl_faces[0], data['pts_d']).float().unsqueeze(-1)

        return data

class CAPEDataModule(pl.LightningDataModule):

    def __init__(self, opt, **kwargs):
        super().__init__()
        self.opt = opt

    def setup(self, stage=None):

        if stage == 'fit':
            self.dataset_train = CAPEDataSet(dataset_path=self.opt.dataset_path,
                                             subject=self.opt.subject,
                                             clothing=self.opt.clothing)

        self.dataset_val = CAPEDataSet(dataset_path=self.opt.dataset_path,
                                        subject=self.opt.subject,
                                        clothing=self.opt.clothing)

        self.meta_info = self.dataset_val.meta_info


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
                                batch_size=self.opt.batch_size,
                                num_workers=self.opt.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.dataset_val,
                                batch_size=1,
                                num_workers=self.opt.num_workers, 
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True)
        return dataloader
