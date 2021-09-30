# load libraries
import trimesh
import torch
import json
import os
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt

import numpy as np
from types import SimpleNamespace
from utils import render
from shapenet_dataloader import ShapeNetMesh, FixedPointsCachedDataset
from shapeflow.layers.deformation_layer import NeuralFlowDeformer
from shapenet_embedding import LatentEmbedder
import shapeflow.utils.train_utils as utils
from torch.utils.data import DataLoader
import pickle
import time
from utils import render


def export_obj_cpu(filename, pc, colors=None, random_trans=[0,0,0]):
    # random_trans = random.uniform(1, 2)
    with open('%s'%(filename), 'w') as f:
        for i,p in enumerate(pc):
            x,y,z = p
            x += random_trans[0]
            y += random_trans[1]
            z += random_trans[2]
            r,g,b = [1,0,0]
            if colors is not None:
                r,g,b = colors[i]
            f.write('v {:.4f} {:.4f} {:.4f} \
                    {:.4f} {:.4f} {:.4f} \n'.format(x, y, z, r, g, b))


# choice of checkpoint to load
run_dir = "/media/andy/Elements/Shapeflow_data/runs/pretrained_ckpt"
checkpoint = "checkpoint_latest.pth.tar_deepdeform_100.pth.tar"
device = torch.device("cuda")


# load training args
args = SimpleNamespace(**json.load(open(os.path.join(run_dir, 'params.json'), 'r')))

# setup model
deformer = NeuralFlowDeformer(latent_size=args.lat_dims, f_width=args.deformer_nf, s_nlayers=2, 
                              s_width=5, method=args.solver, nonlinearity=args.nonlin, arch='imnet',
                              adjoint=args.adjoint, rtol=args.rtol, atol=args.atol, via_hub=True,
                              no_sign_net=(not args.sign_net), symm_dim=(2 if args.symm else None))
lat_params = torch.nn.Parameter(torch.randn(4746, args.lat_dims)*1e-1, requires_grad=True)
deformer.add_lat_params(lat_params)
deformer.to(device)

# load checkpoint
resume_dict = torch.load(os.path.join(run_dir, checkpoint))
start_ep = resume_dict["epoch"]
global_step = resume_dict["global_step"]
tracked_stats = resume_dict["tracked_stats"]
deformer.load_state_dict(resume_dict["deformer_state_dict"])
sample_points = 300
# dataloader
data_root = args.data_root.replace('shapenet_watertight', 'shapenet_simplified')
mesh_dataset = ShapeNetMesh(data_root=data_root, split="train", category='chair', 
                            normals=False)
point_dataset = FixedPointsCachedDataset("/media/andy/Elements/Shapeflow_data/data/shapenet_pointcloud/train/03001627.pkl", npts=sample_points)


# take a sample point cloud from a shape
p = pickle.load(open("/media/andy/Elements/Shapeflow_data/data/shapenet_pointcloud/val/03001627.pkl", "rb"))
name = list(p.keys())[2]
input_points = p[name]
mesh_gt = trimesh.load("/media/andy/Elements/Shapeflow_data/data/shapenet_simplified/val/03001627/bcc73b8ff332b4df3d25ee35360a1f4d/model.ply")

# view point
eye_1 = [.8, .4, .5]
eye_2 = [.3, .4, .9]
center = [0, 0, 0]
up = [0, 1, 0]

def rgb2rgba(rgb):
    """remove white background."""
    rgb = rgb.copy() / 255.
    alpha = np.linalg.norm(1-rgb, axis=-1) != 0
    alpha = alpha.astype(np.float32)[..., None]
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return rgba

# subsample points
point_subsamp = mesh_gt.sample(sample_points)
export_obj_cpu('inputs_fullpc.obj',mesh_gt.sample(2048),random_trans=[-3,0,0])

# img_mesh, _, _, _ = render.render_trimesh(mesh_gt, eye_1, center, up, light_intensity=3)
# img_pt_sub, _, _, _ = render.render_trimesh(trimesh.PointCloud(point_subsamp), 
#                                             eye_1, center, up, light_intensity=3, point_size=8)
# # virtual scan (view 2) and unproject depth
# _, scan_depth, world2cam, cam2img = render.render_trimesh(mesh_gt, eye_2, center, up, res=(112, 112))
# points_unproj = render.unproject_depth_img(scan_depth, cam2img, world2cam)
# img_pt_dep, _, _, _ = render.render_trimesh(trimesh.PointCloud(points_unproj), 
#                                             eye_1, center, up, light_intensity=3, point_size=5)

# size_per_fig = 8
# fig, axes = plt.subplots(figsize=(size_per_fig*4, size_per_fig), ncols=4)
# axes[0].imshow(rgb2rgba(img_mesh))
# axes[0].axis('off')
# # axes[0].set_title("Ground Truth Mesh")

# axes[1].imshow(rgb2rgba(img_pt_sub))
# axes[1].axis('off')
# # axes[1].set_title("Sparse Point Samples")

# d = scan_depth.copy()
# d[scan_depth==0] = np.nan
# axes[2].imshow(d, cmap='coolwarm')
# axes[2].axis('off')
# # axes[2].set_title("Depth Scan")

# axes[3].imshow(rgb2rgba(img_pt_dep))
# axes[3].axis('off')
# # axes[3].set_title("Scanned Points (view 1)")

# plt.show()

embedder = LatentEmbedder(point_dataset, mesh_dataset, deformer, topk=5)

# inputs = input_points[:2048] 
# inputs = points_unproj
inputs = mesh_gt.sample(sample_points) + np.random.randn(sample_points, 3) * 0.005
print(inputs.shape)
export_obj_cpu('inputs_subsampled.obj',inputs,random_trans=[-1.5,0,0])
exit()
input_pts = torch.tensor(inputs)[None].to(device)
lat_codes_pre, lat_codes_post = embedder.embed(input_pts, matching="two_way", verbose=True, lr=1e-2, embedding_niter=30, finetune_niter=30, bs=8, seed=1)

# retrieve, save results
deformed_meshes, orig_meshes, dist = embedder.retrieve(lat_codes_post, tar_pts=inputs, matching="two_way")

asort = np.argsort(dist)
dist = [dist[i] for i in asort]
deformed_meshes_ = [deformed_meshes[i] for i in asort]
orig_meshes_ = [orig_meshes[i] for i in asort]

# pick_idx = np.argmin(dist)
for pick_idx in range(5):
    v, f = deformed_meshes_[pick_idx]
    mesh = trimesh.Trimesh(v, f)
    vo, fo = orig_meshes_[pick_idx]
    mesh_o = trimesh.Trimesh(vo, fo)
    # img_orig, _, _, _ = render.render_trimesh(mesh_o.copy(), eye_1, center, up, res=(512,512), light_intensity=8)
    colors = np.zeros_like(inputs[:sample_points]); colors[:, 1] = 1.;
    export_obj_cpu("latent-opt_deformed_%d.obj"%(pick_idx), v,random_trans=[pick_idx*1.5,0,0])
    export_obj_cpu("latent-opt_orig_%d.obj"%(pick_idx), vo,random_trans=[pick_idx*1.5,2,0])
    # img_def, _, _, _ = render.render_trimesh([mesh.copy(),
    #                                          ],#trimesh.PointCloud(inputs[:512], colors=colors)], 
    #                                          eye_1, center, up, res=(512,512), light_intensity=8,
    #                                          point_size=5)
    # img_gt, _, _, _ = render.render_trimesh(mesh_gt.copy(), eye_1, center, up, res=(512,512), light_intensity=8)
    # fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(24, 8))
    # best = " (best)" if pick_idx == np.argmin(dist) else ""
    # axes[0].imshow(rgb2rgba(img_orig))
    # axes[0].axis('off')
    # axes[0].set_title("Retrieved Shape"+best)
    # axes[1].imshow(rgb2rgba(img_def))
    # axes[1].axis('off')
    # axes[1].set_title("Deformed Shape"+best)
    # axes[2].imshow(rgb2rgba(img_gt))
    # axes[2].axis('off')
    # axes[2].set_title("GT Shape"+best)
    # plt.axis('off')
    # plt.show()

lat_codes_ = torch.tensor(lat_codes_post).to(embedder.device)
lat_src = torch.zeros_like(lat_codes_)
lat_src_tar = torch.stack([lat_src, lat_codes_], dim=1)
_ = embedder.deformer.net.update_latents(lat_src_tar)

# create query grid
r0, r1, r2 = 6, 11, 6
b = mesh.bounding_box.bounds
s = 0.05
xyz_grid = torch.stack(torch.meshgrid(torch.linspace(b[0,0]-s, b[1,0]+s, r0),
                                      torch.linspace(b[0,1]-s, b[1,1]+s, r1),
                                      torch.linspace(b[0,2]-s, b[1,2]+s, r2)), dim=-1)
xyz_pt = xyz_grid.reshape(1, -1, 3).to(embedder.device)
vel = embedder.deformer.net(torch.tensor(0.5), xyz_pt)
vel_np = vel.detach().cpu().numpy().reshape(r0, r1, r2, 3)
xyz_np = xyz_pt.detach().cpu().numpy().reshape(r0, r1, r2, 3)

# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import matplotlib.pyplot as plt
# import numpy as np

# def set_axes_equal(ax):
#     '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
#     cubes as cubes, etc..  This is one possible solution to Matplotlib's
#     ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

#     Input
#       ax: a matplotlib axis, e.g., as output from plt.gca().
#     '''

#     x_limits = [-.2, .2] # ax.get_xlim3d()
#     y_limits = [-.2, .2] # ax.get_ylim3d()
#     z_limits = [-.5, .5] # ax.get_zlim3d()

#     x_range = abs(x_limits[1] - x_limits[0])
#     x_middle = np.mean(x_limits)
#     y_range = abs(y_limits[1] - y_limits[0])
#     y_middle = np.mean(y_limits)
#     z_range = abs(z_limits[1] - z_limits[0])
#     z_middle = np.mean(z_limits)

#     # The plot bounding box is a sphere in the sense of the infinity
#     # norm, hence I call half the max range the plot radius.
#     plot_radius = 0.5*max([x_range, y_range, z_range])

#     ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
#     ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
#     ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection='3d')
# ax.view_init(elev=30, azim=-30)

# v = mesh.copy().vertices
# xyz = xyz_np.reshape(-1, 3)
# uvw = vel_np.reshape(-1, 3)

# ax.plot_trisurf(v[:, 0], v[:, 2], v[:, 1], triangles=mesh.faces, color=np.ones(3), linewidth=0.2)
# ax.quiver(xyz[:, 0], xyz[:, 2], xyz[:, 1],
#           uvw[:, 0], uvw[:, 2], uvw[:, 1],
#           length=0.05, color="black", normalize=True)

# ax.set_axis_off()
# set_axes_equal(ax)
# # plt.savefig("flow.pdf")

