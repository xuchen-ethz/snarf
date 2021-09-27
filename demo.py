
import os
import glob
import hydra
import torch
import imageio
import numpy as np
import pytorch_lightning as pl

from tqdm import trange
from lib.snarf_model import SNARFModel
from lib.model.helpers import rectify_pose

@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())
    pl.seed_everything(42, workers=True)

    # set up model
    meta_info = np.load('meta_info.npz')

    if opt.epoch == 'last':
        checkpoint_path = './checkpoints/last.ckpt'
    else:
        checkpoint_path = glob.glob('./checkpoints/epoch=%d*.ckpt'%opt.epoch)[0]

    model = SNARFModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        opt=opt.model, 
        meta_info=meta_info
    ).cuda()
    # use all bones for initialization during testing
    model.deformer.init_bones = np.arange(24)

    # pose format conversion
    smplx_to_smpl = list(range(66)) + [72, 73, 74, 117, 118, 119]  # SMPLH to SMPL

    # load motion sequence
    motion_path =  hydra.utils.to_absolute_path(opt.demo.motion_path)
    if os.path.isdir(motion_path):
        motion_files = sorted(glob.glob(os.path.join(motion_path, '*.npz')))
        smpl_params_all = []
        for f in motion_files:
            f = np.load(f)
            smpl_params = np.zeros(86)
            smpl_params[0], smpl_params[4:76] = 1, f['pose']
            smpl_params = torch.tensor(smpl_params).float().cuda()
            smpl_params_all.append(smpl_params)
        smpl_params_all = torch.stack(smpl_params_all)

    elif '.npz' in motion_path:
        f = np.load(motion_path)
        smpl_params_all = np.zeros( (f['poses'].shape[0], 86) )
        smpl_params_all[:,0] = 1
        if f['poses'].shape[-1] == 72:
            smpl_params_all[:, 4:76] = f['poses']
        elif f['poses'].shape[-1] == 156:
            smpl_params_all[:, 4:76] = f['poses'][:,smplx_to_smpl]

        root_abs = smpl_params_all[0, 4:7].copy()
        for i in range(smpl_params_all.shape[0]):
            smpl_params_all[i, 4:7] = rectify_pose(smpl_params_all[i, 4:7], root_abs)

        smpl_params_all = torch.tensor(smpl_params_all).float().cuda()

    smpl_params_all = smpl_params_all[::opt.demo.every_n_frames]
    # generate data batch
    images = []
    for i in trange(smpl_params_all.shape[0]):

        smpl_params = smpl_params_all[[i]]
        data = model.smpl_server.forward(smpl_params, absolute=True)
        data['smpl_thetas'] = smpl_params[:, 4:76]
        results = model.plot(data, res=opt.demo.resolution, verbose=opt.demo.verbose, fast_mode=opt.demo.fast_mode)

        images.append(results['img_all'])

        if not os.path.exists('images'):
            os.makedirs('images')
        imageio.imwrite('images/%04d.png'%i, results['img_all'])

        if opt.demo.save_mesh:
            if not os.path.exists('meshes'):
                os.makedirs('meshes')
            results['mesh_def'].export('meshes/%04d_def.ply'%i)
            if opt.demo.verbose:
                results['mesh_cano'].export('meshes/%04d_cano.ply'%i)

    imageio.mimsave('%s.mp4'%opt.demo.output_video_name, images)

if __name__ == '__main__':
    main()