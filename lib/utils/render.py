
import numpy as np
import torch
import cv2

from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures


class Renderer():
    def __init__(self, image_size=512):
        super().__init__()

        self.image_size = image_size

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        R = torch.from_numpy(np.array([[-1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., -1.]])).cuda().float().unsqueeze(0)


        t = torch.from_numpy(np.array([[0., 0.3, 5.]])).cuda().float()

        self.cameras = FoVOrthographicCameras(R=R, T=t,device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=100,blur_radius=0)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
        
    def render_mesh(self, verts, faces, colors=None, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = torch.tensor([0,0,1]).float().to(verts.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            # normal
            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            # shading
            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

            # albedo
            if 'a' in mode: 
                assert(colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            # albedo*shading
            if 't' in mode: 
                assert(colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors*shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)

            return  torch.cat(results, axis=1)

image_size = 512
torch.cuda.set_device(torch.device("cuda:0"))
renderer = Renderer(image_size)

def render(verts, faces, colors=None):
    return renderer.render_mesh(verts, faces, colors)

def render_trimesh(mesh, mode='npta'):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    image = renderer.render_mesh(verts, faces, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    return image


def render_joint(smpl_jnts, bone_ids):
    marker_sz = 6
    line_wd = 2

    image = np.ones((image_size, image_size,3), dtype=np.uint8)*255 
    smpl_jnts[:,1] += 0.3
    smpl_jnts[:,1] = -smpl_jnts[:,1] 
    smpl_jnts = smpl_jnts[:,:2]*image_size/2 + image_size/2

    for b in bone_ids:
        if b[0]<0 : continue
        joint = smpl_jnts[b[0]]
        cv2.circle(image, joint.astype('int32'), color=(0,0,0), radius=marker_sz, thickness=-1)

        joint2 = smpl_jnts[b[1]]
        cv2.circle(image, joint2.astype('int32'), color=(0,0,0), radius=marker_sz, thickness=-1)

        cv2.line(image, joint2.astype('int32'), joint.astype('int32'), color=(0,0,0), thickness=int(line_wd))

    return image



def weights2colors(weights):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')

    colors = [ 'pink', #0
                'blue', #1
                'green', #2
                'red', #3
                'pink', #4
                'pink', #5
                'pink', #6
                'green', #7
                'blue', #8
                'red', #9
                'pink', #10
                'pink', #11
                'pink', #12
                'blue', #13
                'green', #14
                'red', #15
                'cyan', #16
                'darkgreen', #17
                'pink', #18
                'pink', #19
                'blue', #20
                'green', #21
                'pink', #22
                'pink' #23
    ]


    color_mapping = {'cyan': cmap.colors[3],
                    'blue': cmap.colors[1],
                    'darkgreen': cmap.colors[1],
                    'green':cmap.colors[3],
                    'pink': [1,1,1],
                    'red':cmap.colors[5],
                    }

    for i in range(len(colors)):
        colors[i] = np.array(color_mapping[colors[i]])

    colors = np.stack(colors)[None]# [1x24x3]
    verts_colors = weights[:,:,None] * colors
    verts_colors = verts_colors.sum(1)
    return verts_colors