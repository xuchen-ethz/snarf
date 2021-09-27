
import os
import glob
import hydra
import torch
import numpy as np
import pytorch_lightning as pl

from lib.snarf_model import SNARFModel

@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(10) 

    datamodule = hydra.utils.instantiate(opt.datamodule, opt.datamodule)
    datamodule.setup(stage='test')

    trainer = pl.Trainer(**opt.trainer)

    if opt.epoch == 'last':
        checkpoint_path = './checkpoints/last.ckpt'
    else:
        checkpoint_path = glob.glob('./checkpoints/epoch=%d*.ckpt'%opt.epoch)[0]

    model = SNARFModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        opt=opt.model, 
        meta_info=datamodule.meta_info
    )
    # use all bones for initialization during testing
    model.deformer.init_bones = np.arange(24)

    results = trainer.test(model, datamodule=datamodule, verbose=True)

    np.savetxt('./results_%s_%s_%s.txt'%(os.path.basename(opt.datamodule.dataset_path),opt.datamodule.subject, str(opt.epoch)), np.array([results[0]['valid_bbox_iou'], results[0]['valid_surf_iou']]))

if __name__ == '__main__':
    main()