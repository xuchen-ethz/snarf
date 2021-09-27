import os
import glob
import argparse
import torch
import trimesh

import numpy as np
import torch.nn.functional as F

from body_model import BodyModel
from utils import export_points

from tqdm import tqdm,trange
from shutil import copyfile

parser = argparse.ArgumentParser('Read and sample AMASS dataset.')
parser.add_argument('--dataset_path', type=str, default='data/',
                    help='Path to AMASS dataset.')

parser.add_argument('--poseprior', action='store_true',
                help='Generate data for posprior dataset')

parser.add_argument('--bm_path', type=str, default='lib/smpl/smpl_model',
                    help='Path to body model')

parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')

parser.add_argument('--output_folder', type=str, default='data/DFaust_processed',
                    help='Output path for points.')

parser.add_argument('--points_size', type=int, default=200000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=.5,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')

parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')

parser.add_argument('--skip', type=int, default=1,
                    help='Take every x frames.')

def process_single_file(vertices, root_orient, pose, joints, bone_transforms, root_loc, frame_name, betas, gender, skinning_weights_vertices, faces, subset, args):
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # Get extents of model.
    bb_min = np.min(vertices, axis=0)
    bb_max = np.max(vertices, axis=0)
    # total_size = np.sqrt(np.square(bb_max - bb_min).sum())
    total_size = (bb_max - bb_min).max()

    # Set the center (although this should usually be the origin already).
    loc = np.array(
        [(bb_min[0] + bb_max[0]) / 2,
         (bb_min[1] + bb_max[1]) / 2,
         (bb_min[2] + bb_max[2]) / 2]
    )
    # Scales all dimensions equally.
    scale = total_size / (1 - args.bbox_padding)
    export_points(body_mesh, subset, frame_name, loc, scale, args, joints=joints, bone_transforms=bone_transforms, root_loc=root_loc, root_orient=root_orient, pose=pose, betas=betas, gender=gender, skinning_weights_vertices=skinning_weights_vertices, vertices=vertices, faces=faces)


def amass_extract(args):
    dfaust_dir = os.path.join(args.dataset_path, 'DFaust_67')
    subjects = [os.path.basename(s_dir) for s_dir in sorted(glob.glob(os.path.join(dfaust_dir, '*')))]

    for subject in tqdm(subjects):
        subject_dir = os.path.join(dfaust_dir, subject)

        shape_data = np.load(os.path.join(subject_dir, 'shape.npz'))

        # Save shape data
        output_shape_folder = os.path.join(args.output_folder, 'shapes')
        if not os.path.exists(output_shape_folder):
            os.makedirs(output_shape_folder)
        copyfile(os.path.join(subject_dir, 'shape.npz'), os.path.join(output_shape_folder, '%s_shape.npz'%subject))

        # Generate and save rest-pose for current subject
        gender = shape_data['gender'].item()
        betas = torch.Tensor(shape_data['betas'][:10]).unsqueeze(0).cuda()
        bm_path = os.path.join(args.bm_path, 'SMPL_%s.pkl'%(gender.upper()))
        bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1).cuda()

        # Get skinning weights
        with torch.no_grad():
            body = bm(betas=betas)
            vertices = body.v.detach().cpu().numpy()[0]
            faces = bm.f.detach().cpu().numpy()

            skinning_weights_vertices = bm.weights
            skinning_weights_vertices = skinning_weights_vertices.detach().cpu().numpy()

        # Read pose sequences
        sequences = []
        if args.poseprior:
            pose_dir = os.path.join(args.dataset_path, 'MPI_Limits', '03099')
            for s_dir in glob.glob(os.path.join(pose_dir, '*.npz')):
                sequence = os.path.basename(s_dir)
                if sequence in ['op2_poses.npz', 'op3_poses.npz', 'op4_poses.npz', 'op5_poses.npz', 'op7_poses.npz', 'op8_poses.npz', 'op9_poses.npz']:
                    sequences.append(sequence)
        else:
            pose_dir = subject_dir
            for s_dir in glob.glob(os.path.join(pose_dir, '*.npz')):
                sequence = os.path.basename(s_dir)
                if sequence not in ['shape.npz']:
                    sequences.append(sequence)

        for sequence in tqdm(sequences):
            sequence_path = os.path.join(pose_dir, sequence)
            sequence_name = sequence[:]
            data = np.load(sequence_path, allow_pickle=True)

            poses = data['poses'][::args.skip]
            trans = data['trans'][::args.skip]

            batch_size = poses.shape[0]
            bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=batch_size).cuda()
            faces = bm.f.detach().cpu().numpy()

            pose_body = torch.Tensor(poses[:, 3:66]).cuda()
            pose_hand = torch.Tensor(poses[:, 66:72]).cuda()
            pose = torch.Tensor(poses[:, :72]).cuda()
            root_orient = torch.Tensor(poses[:, :3]).cuda()
            trans = torch.zeros(batch_size, 3, dtype=torch.float32).cuda()
        
            with torch.no_grad():

                body = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, betas=betas.expand(batch_size,-1), trans=trans)

                trans_ = F.pad(trans, [0, 1]).view(batch_size, 1, -1, 1)
                trans_ = torch.cat([torch.zeros(batch_size, 1, 4, 3, device=trans_.device), trans_], dim=-1)
                bone_transforms = body.bone_transforms + trans_
                bone_transforms = torch.inverse(bone_transforms).detach().cpu().numpy()
                bone_transforms = bone_transforms[:, :, :]

                pose_body = pose_body.detach().cpu().numpy()
                pose = pose.detach().cpu().numpy()
                joints = body.Jtr.detach().cpu().numpy()
                vertices = body.v.detach().cpu().numpy()
                trans = trans.detach().cpu().numpy()
                root_orient = root_orient.detach().cpu().numpy()

            for f_idx in trange(batch_size):
                frame_name = sequence_name + '_{:06d}'.format(f_idx)
                process_single_file(vertices[f_idx], 
                                    root_orient[f_idx], 
                                    pose[f_idx], 
                                    joints[f_idx], 
                                    bone_transforms[f_idx], 
                                    trans[f_idx], 
                                    frame_name, 
                                    betas[0].detach().cpu().numpy(), 
                                    gender, 
                                    skinning_weights_vertices, 
                                    faces, 
                                    subject,
                                    args)

def main(args):
    amass_extract(args)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
