import os
import hydra
import torch
import wandb
import imageio
import numpy as np
import pytorch_lightning as pl

from lib.model.smpl import SMPLServer
from lib.model.sample import PointOnBones
from lib.model.network import ImplicitNetwork
from lib.model.metrics import calculate_iou
from lib.utils.meshing import generate_mesh
from lib.model.helpers import masked_softmax
from lib.model.deformer import ForwardDeformer, skinning
from lib.utils.render import render_trimesh, render_joint, weights2colors

class SNARFModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.opt = opt

        self.network = ImplicitNetwork(**opt.network)
        self.deformer = ForwardDeformer(opt.deformer)

        print(self.network)
        print(self.deformer)

        gender      = str(meta_info['gender'])
        betas       = meta_info['betas'] if 'betas' in meta_info else None
        v_template  = meta_info['v_template'] if 'v_template' in meta_info else None

        self.smpl_server = SMPLServer(gender=gender, betas=betas, v_template=v_template)
        self.smpl_faces  = torch.tensor(self.smpl_server.smpl.faces.astype('int')).unsqueeze(0).cuda()
        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        self.data_processor = data_processor

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt.optim.lr)
        return optimizer

    def forward(self, pts_d, smpl_tfs, smpl_thetas, eval_mode=True):

        # rectify rest pose 
        smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs, self.smpl_server.tfs_c_inv)

        cond = {'smpl': smpl_thetas[:,3:]/np.pi}
        
        batch_points = 60000

        accum_pred = []
        # split to prevent out of memory
        for pts_d_split in torch.split(pts_d, batch_points, dim=1):

            # compute canonical correspondences
            pts_c, intermediates = self.deformer(pts_d_split, cond, smpl_tfs, eval_mode=eval_mode)

            # query occuancy in canonical space
            num_batch, num_point, num_init, num_dim = pts_c.shape
            pts_c = pts_c.reshape(num_batch, num_point * num_init, num_dim)
            occ_pd = self.network(pts_c, cond).reshape(num_batch, num_point, num_init)

            # aggregate occupancy probablities
            mask = intermediates['valid_ids']
            if eval_mode:
                occ_pd = masked_softmax(occ_pd, mask, dim=-1, mode='max')
            else:
                occ_pd = masked_softmax(occ_pd, mask, dim=-1, mode='softmax', soft_blend=self.opt.soft_blend)

            accum_pred.append(occ_pd)

        accum_pred = torch.cat(accum_pred, 1)   

        return accum_pred

    def training_step(self, data, data_idx):

        # Data prep
        if self.data_processor is not None:
            data = self.data_processor.process(data)

        # BCE loss
        occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_thetas'], eval_mode=False)
        loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(occ_pd, data['occ_gt'])
        self.log('train_bce', loss_bce)
        loss = loss_bce

        # Bootstrapping
        num_batch = data['pts_d'].shape[0]
        cond = {'smpl': data['smpl_thetas'][:,3:]/np.pi}

        # Bone occupancy loss
        if self.current_epoch < self.opt.nepochs_pretrain:
            if self.opt.lambda_bone_occ > 0:

                pts_c, occ_gt = self.sampler_bone.get_points(self.smpl_server.joints_c.expand(num_batch, -1, -1))
                occ_pd = self.network(pts_c, cond)
                loss_bone_occ = torch.nn.functional.binary_cross_entropy_with_logits(occ_pd, occ_gt.unsqueeze(-1))

                loss = loss + self.opt.lambda_bone_occ * loss_bone_occ
                self.log('train_bone_occ', loss_bone_occ)

            # Joint weight loss
            if self.opt.lambda_bone_w > 0:

                pts_c, w_gt = self.sampler_bone.get_joints(self.smpl_server.joints_c.expand(num_batch, -1, -1))
                w_pd = self.deformer.query_weights(pts_c, cond)
                loss_bone_w = torch.nn.functional.mse_loss(w_pd, w_gt)

                loss = loss + self.opt.lambda_bone_w * loss_bone_w
                self.log('train_bone_w', loss_bone_w)

        return loss
    
    def validation_step(self, data, data_idx):

        if self.data_processor is not None:
            data = self.data_processor.process(data)

        with torch.no_grad():
            if data_idx == 0:
                img_all = self.plot(data)['img_all']
                self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
                
            occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_thetas'], eval_mode=True)

            _, num_point, _ = data['occ_gt'].shape
            bbox_iou = calculate_iou(data['occ_gt'][:,:num_point//2]>0.5, occ_pd[:,:num_point//2]>0)
            surf_iou = calculate_iou(data['occ_gt'][:,num_point//2:]>0.5, occ_pd[:,num_point//2:]>0)

        return {'bbox_iou':bbox_iou, 'surf_iou':surf_iou}

    def validation_epoch_end(self, validation_step_outputs):

        bbox_ious, surf_ious = [], []
        for output in validation_step_outputs:
            bbox_ious.append(output['bbox_iou'])
            surf_ious.append(output['surf_iou'])
        
        self.log('valid_bbox_iou', torch.stack(bbox_ious).mean())
        self.log('valid_surf_iou', torch.stack(surf_ious).mean())

    def test_step(self, data, data_idx):

        with torch.no_grad():

            occ_pd = self.forward(data['pts_d'], data['smpl_tfs'], data['smpl_thetas'], eval_mode=True)

            _, num_point, _ = data['occ_gt'].shape
            bbox_iou = calculate_iou(data['occ_gt'][:,:num_point//2]>0.5, occ_pd[:,:num_point//2]>0)
            surf_iou = calculate_iou(data['occ_gt'][:,num_point//2:]>0.5, occ_pd[:,num_point//2:]>0)

        return {'bbox_iou':bbox_iou, 'surf_iou':surf_iou}
            
    def test_epoch_end(self, test_step_outputs):
        return self.validation_epoch_end(test_step_outputs)

    def plot(self, data, res=128, verbose=True, fast_mode=False):

        res_up = np.log2(res//32)

        if verbose:
            surf_pred_cano = self.extract_mesh(self.smpl_server.verts_c, data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=True, with_weights=True)
            surf_pred_def = self.extract_mesh(data['smpl_verts'][[0]], data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=False, with_weights=False)

            img_pred_cano = render_trimesh(surf_pred_cano)
            img_pred_def  = render_trimesh(surf_pred_def)
            
            img_joint = render_joint(data['smpl_jnts'].data.cpu().numpy()[0],self.smpl_server.bone_ids)
            img_pred_def[1024:,:,:3] = 255
            img_pred_def[1024:-512,:, :3] = img_joint
            img_pred_def[1024:-512,:, -1] = 255

            results = {
                'img_all': np.concatenate([img_pred_cano, img_pred_def], axis=1),
                'mesh_cano': surf_pred_cano,
                'mesh_def' : surf_pred_def
            }
        else:
            smpl_verts = self.smpl_server.verts_c if fast_mode else data['smpl_verts'][[0]]

            surf_pred_def = self.extract_mesh(smpl_verts, data['smpl_tfs'][[0]], data['smpl_thetas'][[0]], res_up=res_up, canonical=False, with_weights=False, fast_mode=fast_mode)
                        
            img_pred_def  = render_trimesh(surf_pred_def, mode='p')
            results = {
                'img_all': img_pred_def,
                'mesh_def' : surf_pred_def
            }
        

        return results    

    def extract_mesh(self, smpl_verts, smpl_tfs, smpl_thetas, canonical=False, with_weights=False, res_up=2, fast_mode=False):
        '''
        In fast mode, we extract canonical mesh and then forward skin it to posed space.
        This is faster as it bypasses root finding.
        However, it's not deforming the continuous field, but the discrete mesh.
        '''
        if canonical or fast_mode:
            occ_func = lambda x: self.network(x, {'smpl': smpl_thetas[:,3:]/np.pi}).reshape(-1, 1)
        else:
            occ_func = lambda x: self.forward(x, smpl_tfs, smpl_thetas, eval_mode=True).reshape(-1, 1)
            
        mesh = generate_mesh(occ_func, smpl_verts.squeeze(0),res_up=res_up)


        if fast_mode:
            verts  = torch.tensor(mesh.vertices).type_as(smpl_verts)
            weights = self.deformer.query_weights(verts[None], None).clamp(0,1)[0]

            smpl_tfs = torch.einsum('bnij,njk->bnik', smpl_tfs, self.smpl_server.tfs_c_inv)
            
            verts_mesh_deformed = skinning(verts.unsqueeze(0), weights.unsqueeze(0), smpl_tfs).data.cpu().numpy()[0]
            mesh.vertices = verts_mesh_deformed

        if with_weights:
            verts  = torch.tensor(mesh.vertices).cuda().float()
            weights = self.deformer.query_weights(verts[None], None).clamp(0,1)[0]
            mesh.visual.vertex_colors = weights2colors(weights.data.cpu().numpy())

        return mesh
