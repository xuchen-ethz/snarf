
import pytorch_lightning as pl
import hydra
import torch
import yaml
import os
import numpy as np

from lib.snarf_model import SNARFModel

@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())

    pl.seed_everything(42, workers=True)

    torch.set_num_threads(10)

    # dataset
    datamodule = hydra.utils.instantiate(opt.datamodule, opt.datamodule)
    datamodule.setup(stage='fit')
    np.savez('meta_info.npz', **datamodule.meta_info)

    data_processor = None
    if 'processor' in opt.datamodule:
        data_processor = hydra.utils.instantiate(opt.datamodule.processor,
                                                 opt.datamodule.processor,
                                                 meta_info=datamodule.meta_info)

    # logger
    with open('.hydra/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger = pl.loggers.WandbLogger(project='snarf', config=config)

    # checkpoint
    checkpoint_path = './checkpoints/last.ckpt'
    if not os.path.exists(checkpoint_path) or not opt.resume:
        checkpoint_path = None 

    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1,
                                                        monitor=None, 
                                                        dirpath='./checkpoints',
                                                        save_last=True,
                                                        every_n_val_epochs=1)


    trainer = pl.Trainer(logger=logger, 
                        callbacks=[checkpoint_callback],
                        accelerator=None,
                        resume_from_checkpoint=checkpoint_path,
                        **opt.trainer)

    model = SNARFModel(opt=opt.model, 
                    meta_info=datamodule.meta_info,
                    data_processor=data_processor)

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()