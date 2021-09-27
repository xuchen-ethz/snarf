import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import numpy as np


def export_points(mesh, subject, modelname, loc, scale, args, **kwargs):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % modelname)
        return

    kwargs_new = {}
    for k, v in kwargs.items():
        if v is not None:
            kwargs_new[k] = v

    filename = os.path.join(args.output_folder, 'points', subject, modelname + '.npz')

    if not args.overwrite and os.path.exists(filename):
        print('Points already exist: %s' % filename)
        return

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    # Scale points in (padded) unit box back to the original space
    points_uniform *= scale
    points_uniform += np.expand_dims(loc, axis=0)
    # Sample points around mesh surface
    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)
    points = torch.tensor(points).cuda().float().unsqueeze(0)
    vertices = torch.tensor(kwargs['vertices']).cuda().float().unsqueeze(0)
    faces = torch.tensor(kwargs['faces'], dtype=torch.int64).cuda()

    import kaolin
    occupancies = kaolin.ops.mesh.check_sign(vertices, faces, points)


    points = points.cpu().numpy()
    occupancies = occupancies.cpu().numpy()

    # Compress
    if args.float16:
        dtype = np.float16
    else:
        dtype = np.float32

    points = points.astype(dtype)

    if args.packbits:
        occupancies = np.packbits(occupancies)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    # print('Writing points: %s' % filename)
    np.savez(filename, points=points, occupancies=occupancies,
             loc=loc, scale=scale,
             **kwargs_new)
