#!/usr/bin/env python
# coding: utf-8

# ## Load deformer and visualize results

# In[1]:


# load libraries
import trimesh
import torch
import json
import os
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import numpy as np
from types import SimpleNamespace
from utils import render
from shapenet_dataloader import ShapeNetMesh
from shapeflow.layers.deformation_layer import NeuralFlowDeformer


# ### Options

# In[2]:


# choice of checkpoint to load
# run_dir = "runs/symm_lat256_nosign_b32"  # "runs/pretrained_ckpt"
run_dir = "/media/andy/Elements/Shapeflow_data/runs/pretrained_ckpt"
# run_dir = "runs/pretrained_ckpt"
checkpoint = "checkpoint_latest.pth.tar_deepdeform_100.pth.tar"  # "checkpoint_latest.pth.tar_shapeflow_034.pth.tar" #  "checkpoint_latest.pth.tar_shapeflow_100.pth.tar"
device = torch.device("cuda")


# ### Setup

# In[19]:


# load training args
args = SimpleNamespace(**json.load(open(os.path.join(run_dir, 'params.json'), 'r')))
if not 'symm' in args.__dict__.keys():
    args.symm = None

# setup model
deformer = NeuralFlowDeformer(latent_size=args.lat_dims, f_width=args.deformer_nf, s_nlayers=2, 
                              s_width=5, method=args.solver, nonlinearity=args.nonlin, arch='imnet',
                              adjoint=args.adjoint, rtol=args.rtol, atol=args.atol, via_hub=True,
                              no_sign_net=(not args.sign_net), symm_dim=(2 if args.symm else None))
lat_params = torch.nn.Parameter(torch.randn(4746, args.lat_dims)*1e-1, requires_grad=True)
deformer.add_lat_params(lat_params)
print(device)
deformer.to(device)

# load checkpoint
resume_dict = torch.load(os.path.join(run_dir, checkpoint))
start_ep = resume_dict["epoch"]
global_step = resume_dict["global_step"]
tracked_stats = resume_dict["tracked_stats"]
deformer.load_state_dict(resume_dict["deformer_state_dict"])

# dataloader
data_root = args.data_root.replace('shapenet_watertight', 'shapenet_simplified')
dset = ShapeNetMesh(data_root=data_root, split="train", category="chair", 
                    normals=False)

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

def deformation_two_shapes():
    # ### Deformation between two shapes

    # Rendering view point
    eye_1 = [.8, .4, .5]
    d_eye = np.linalg.norm(eye_1)
    eye_2 = np.array(eye_1) / d_eye * 2
    center = [0, 0, 0]
    up = [0, 1, 0]

    # pick source and target (between 0 and 4745)
    source_idx = 0
    target_idx = 1

    # get the corresponding latent codes
    source_latent = deformer.get_lat_params(source_idx)[None]   # unsqueeze batch dimension
    target_latent = deformer.get_lat_params(target_idx)[None]   # unsqueeze batch dimension
    zero_latent = torch.zeros_like(source_latent)  # "hub" latent
    print(f"Source latent shape: {source_latent.shape}")
    print(f"Target latent shape: {target_latent.shape}")


    # access all latent parameters
    all_latents = deformer.net.lat_params
    print(f"Shape of all latent params: {all_latents.shape}")


    # get the source and target shapes (meshes)
    v_source, f_source = dset.get_single(source_idx)
    v_target, f_target = dset.get_single(target_idx)
    v_source = v_source[None].to(device)  # unsqueeze batch dimension
    v_target = v_target[None].to(device)  # unsqueeze batch dimension
    print(f"Source v shape: {v_source.shape}; f shape: {f_source.shape}")
    print(f"Target v shape: {v_target.shape}; f shape: {f_target.shape}")


    # deform source to target
    lat_path = lambda l_src_, l_tar_: torch.stack([l_src_, zero_latent, l_tar_], dim=1)  # alias
    with torch.no_grad():  # no need grad for inference
        v_src_to_tar = deformer(v_source, lat_path(source_latent, target_latent))[0]  # source to target
        v_tar_to_src = deformer(v_target, lat_path(target_latent, source_latent))[0]  # target to source
    print(f"Source to target v shape: {v_src_to_tar.shape}; f shape: {f_source.shape}")
    print(f"Target to target v shape: {v_tar_to_src.shape}; f shape: {f_target.shape}")


    # convert to numpy
    v_src_to_tar = v_src_to_tar.detach().cpu().numpy()
    v_tar_to_src = v_tar_to_src.detach().cpu().numpy()
    v_src = v_source[0].detach().cpu().numpy()
    v_tar = v_target[0].detach().cpu().numpy()
    f_src = f_source.detach().cpu().numpy()
    f_tar = f_target.detach().cpu().numpy()

    export_obj_cpu('v_src_to_tar.obj',v_src_to_tar, random_trans=[-1.5,0,0])
    export_obj_cpu('v_tar_to_src.obj',v_tar_to_src, random_trans=[1.5,0,0])
    export_obj_cpu('v_src.obj',v_src, random_trans=[-4,0,0])
    export_obj_cpu('v_tar.obj',v_tar, random_trans=[4,0,0])
    export_obj_cpu('f_src.obj',f_src, random_trans=[0,0,0])
    export_obj_cpu('f_tar.obj',f_tar, random_trans=[0,0,0])

    # # render and visualize
    # img_src, _, _, _ = render.render_trimesh(trimesh.Trimesh(v_src, f_src), eye_1, center, up, res=(224,224))
    # img_tar, _, _, _ = render.render_trimesh(trimesh.Trimesh(v_tar, f_tar), eye_1, center, up, res=(224,224))
    # img_src_to_tar, _, _, _ = render.render_trimesh(trimesh.Trimesh(v_src_to_tar, f_src), 
    #                                                 eye_1, center, up, res=(224,224))
    # img_tar_to_src, _, _, _ = render.render_trimesh(trimesh.Trimesh(v_tar_to_src, f_tar), 
    #                                                 eye_1, center, up, res=(224,224))

    # # plot renderings
    # fig, axes = plt.subplots(ncols=2, nrows=2)
    # axes[0, 0].imshow(img_src)
    # axes[1, 0].imshow(img_tar)
    # axes[0, 1].imshow(img_src_to_tar)
    # axes[1, 1].imshow(img_tar_to_src)
    # plt.show()

def visualize_deformation_n_shapes():
    # ### Visualize deformation between n shapes

    np.random.seed(3)
    shape_indices = np.random.choice(4745, 5)  # choose between [0, 4745]

    n = len(shape_indices)
    imgs = [[None for _ in range(n)] for i in range(n)]
    imgs_src = [None for _ in range(n)]
    imgs_tar = [None for _ in range(n)]
    faces = [None for _ in range(n)]
    verts = [None for _ in range(n)]

    for i, idx in enumerate(shape_indices):
        v, f = dset.get_single(idx)
        verts[i] = v[None].to(device)
        faces[i] = f.detach().cpu().numpy()
        
    # deform all pairs
    l_zero = torch.zeros_like(deformer.get_lat_params(0)[None] )
    lat_path = lambda l_src_, l_tar_: torch.stack([l_src_, l_zero, l_tar_], dim=1)
    ten2npy = lambda tensor: tensor.detach().cpu().numpy()

    pbar = tqdm(total=n**2)
    tqdm.write("computing pairwise deformations...")

    with torch.no_grad():
        for i, ii in enumerate(shape_indices):
            for j, jj in enumerate(shape_indices):
                # get the latent codes corresponding to these shapes
                l_src = deformer.get_lat_params(ii)[None] 
                l_tar = deformer.get_lat_params(jj)[None]

                # deform src to target
                v_s2t = deformer(verts[i], lat_path(l_src, l_tar))[0]  # source to target
                if i == j:
                    color = [0., 1., 0.]
                else:
                    color = [1., 1., 1.]
                mesh_s2t = trimesh.Trimesh(ten2npy(v_s2t), faces[i])
                mesh_s2t.visual.vertex_colors = np.array(color)
                img, _, _, _ = render.render_trimesh(mesh_s2t, 
                                                     eye_1, center, up, res=(224,224))
                imgs[i][j] = img
                pbar.update(1)
    pbar.close()


    # render colored imgs for source and target
    for i in range(n):
        m_src = trimesh.Trimesh(ten2npy(verts[i][0]), faces[i])
        m_tar = trimesh.Trimesh(ten2npy(verts[i][0]), faces[i])
        m_src.visual.vertex_colors = np.array([1., 0.3, 0.3])
        m_tar.visual.vertex_colors = np.array([0.3, 0.3, 1.])
        img_src, _, _, _ = render.render_trimesh(m_src, eye_1, center, up, res=(224,224), light_intensity=4, ambient_intensity=.2)
        img_tar, _, _, _ = render.render_trimesh(m_tar, eye_1, center, up, res=(224,224), light_intensity=4, ambient_intensity=.2)
        imgs_src[i] = img_src
        imgs_tar[i] = img_tar

        
    # plot transformation table
    fig, axes = plt.subplots(figsize=(3*n, 3*n), nrows=n+1, ncols=n+1)
    for i in range(0, n+1):
        for j in range(0, n+1):
            if i == 0 and j > 0:
                axes[i, j].imshow(imgs_tar[j-1])
            if i > 0 and j == 0:
                axes[i, j].imshow(imgs_src[i-1])
            if i > 0 and j > 0:
                axes[i, j].imshow(imgs[i-1][j-1])
            axes[i, j].axis('off')
    plt.show()


def visualize_deformation_mean_shapes():
    # ## Visualize deformation to mean shapes (hub)

    np.random.seed(1)
    shape_indices = np.random.choice(4745, 5)  # choose between [0, 4745]
    n = len(shape_indices)
    imgs_src = [None for _ in range(n)]
    imgs_tar = [None for _ in range(n)]
    faces = [None for _ in range(n)]
    verts = [None for _ in range(n)]  # verts tensor
    mesh_o = [None for _ in range(n)]  # mesh original
    mesh_d = [None for _ in range(n)]  # mesh deformed

    for i, idx in enumerate(shape_indices):
        v, f = dset.get_single(idx)
        verts[i] = v[None].to(device)
        faces[i] = f.detach().cpu().numpy()

    lat_path = lambda l_src_, l_tar_: torch.stack([l_src_, l_tar_], dim=1)
    ten2npy = lambda tensor: tensor.detach().cpu().numpy()

    pbar = tqdm(total=n)

    with torch.no_grad():
        for i, ii in enumerate(shape_indices):
            # get the latent codes corresponding to these shapes
            l_src = deformer.get_lat_params(ii)[None] 
            l_tar = torch.zeros_like(l_src)  # target is the "hub"

            # deform source to target
            v_s2t = deformer(verts[i], lat_path(l_src, l_tar))[0]  # source to target
            mesh_d[i] = trimesh.Trimesh(ten2npy(v_s2t), faces[i], process=False)
            mesh_o[i] = trimesh.Trimesh(ten2npy(verts[i][0]), faces[i], process=False)
            pbar.update(1)
    pbar.close()


    # In[7]:


    # colorize vertices based on mean shape coordinates
    bboxes_d = [m.bounding_box.bounds for m in mesh_d]
    bboxes_d = np.concatenate(bboxes_d, axis=0)
    bboxes_d = np.stack([np.min(bboxes_d, axis=0), np.max(bboxes_d, axis=0)], axis=0)

    for idx, (mo, md) in enumerate(zip(mesh_o, mesh_d)):
        color = (md.vertices.copy() - bboxes_d[0]) / (bboxes_d[1] - bboxes_d[0])
        md.visual.vertex_colors = color.copy()
        mo.visual.vertex_colors = color.copy()
        md_ = md.copy()#; md_.vertices *= 0.5
        mo_ = mo.copy()
        img_tar, _, _, _ = render.render_trimesh(md_, eye_2, center, up, res=(224, 224))
        img_src, _, _, _ = render.render_trimesh(mo_, eye_1, center, up, res=(224, 224))
        
        imgs_src[idx] = img_src
        imgs_tar[idx] = img_tar
        
    # plot transformation table
    fig, axes = plt.subplots(figsize=(5*n, 10), nrows=2, ncols=n)
    for i in range(n):
        axes[0, i].imshow(imgs_src[i])
        axes[1, i].imshow(imgs_tar[i])
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()


    # In[ ]:


    dirname = "canonicalized_example"
    os.makedirs(dirname, exist_ok=True)
    for idx, (md, mo) in enumerate(zip(mesh_d, mesh_o)):
        _ = md.export(os.path.join(dirname, f"can_{idx}.ply"))
        _ = mo.export(os.path.join(dirname, f"orig_{idx}.ply"))


    # In[ ]:


    for ii, i in enumerate(shape_indices):
        print(f"{ii} "+dset.fnames[i])


def new_observation():
    # load new observation
    mesh_new = trimesh.load("/media/andy/Elements/Shapeflow_data/data/shapenet_simplified/val/03001627/bcc73b8ff332b4df3d25ee35360a1f4d/model.ply")
    # lat_new = np.load("lat_codes_pre_tune.npy")

    np.random.seed(0)
    shape_indices = np.random.choice(4745, 5)  # choose between [0, 4745]
    n = len(shape_indices)
    imgs_src = [None for _ in range(n+1)]
    imgs_tar = [None for _ in range(n+1)]
    faces = [None for _ in range(n+1)]
    verts = [None for _ in range(n+1)]  # verts tensor
    mesh_o = [None for _ in range(n+1)]  # mesh original
    mesh_d = [None for _ in range(n+1)]  # mesh deformed

    for i, idx in enumerate(shape_indices):
        v, f = dset.get_single(idx)
        verts[i] = v[None].to(device)
        faces[i] = f.detach().cpu().numpy()
    verts[-1] = torch.from_numpy(mesh_new.vertices)[None].float().to(device)
    faces[-1] = mesh_new.faces

    lat_path = lambda l_src_, l_tar_: torch.stack([l_src_, l_tar_], dim=1)
    ten2npy = lambda tensor: tensor.detach().cpu().numpy()

    pbar = tqdm(total=n+1)

    with torch.no_grad():
        for i, ii in enumerate(shape_indices):
            # get the latent codes corresponding to these shapes
            l_src = deformer.get_lat_params(ii)[None] 
            l_tar = torch.zeros_like(l_src)  # target is the "hub"

            # deform source to target
            v_s2t = deformer(verts[i], lat_path(l_src, l_tar))[0]  # source to target
            mesh_d[i] = trimesh.Trimesh(ten2npy(v_s2t), faces[i], process=False)
            mesh_o[i] = trimesh.Trimesh(ten2npy(verts[i][0]), faces[i], process=False)
            pbar.update(1)
            
        # new observation
        # l_src = torch.from_numpy(lat_new).to(device)
        # l_tar = torch.zeros_like(l_src)  # target is the "hub"
        # deform source to target
        v_s2t = deformer(verts[-1], lat_path(l_src, l_tar))[0]  # source to target
        mesh_d[-1] = trimesh.Trimesh(ten2npy(v_s2t), faces[-1], process=False)
        mesh_o[-1] = trimesh.Trimesh(ten2npy(verts[-1][0]), faces[-1], process=False)
        pbar.update(1)
            
    pbar.close()

    # colorize vertices based on mean shape coordinates
    bboxes_d = [m.bounding_box.bounds for m in mesh_d]
    bboxes_d = np.concatenate(bboxes_d, axis=0)
    bboxes_d = np.stack([np.min(bboxes_d, axis=0), np.max(bboxes_d, axis=0)], axis=0)

    for idx, (mo, md) in enumerate(zip(mesh_o, mesh_d)):
        color = (md.vertices.copy() - bboxes_d[0]) / (bboxes_d[1] - bboxes_d[0])
        md.visual.vertex_colors = color.copy()
        mo.visual.vertex_colors = color.copy()
        md_ = md.copy()#; md_.vertices *= 0.5
        mo_ = mo.copy()
        export_obj_cpu('md_%s.obj'%(str(idx)), md_.vertices, random_trans=[idx,0,0])
        export_obj_cpu('mo_%s.obj'%(str(idx)), mo_.vertices, random_trans=[idx,2,0])
        # img_tar, _, _, _ = render.render_trimesh(md_, eye_1, center, up, res=(224, 224))
        # img_src, _, _, _ = render.render_trimesh(mo_, eye_1, center, up, res=(224, 224))
        
    #     imgs_src[idx] = img_src
    #     imgs_tar[idx] = img_tar
        
    # # plot transformation table
    # fig, axes = plt.subplots(figsize=(5*(n+1), 10), nrows=2, ncols=(n+1))
    # for ii, i in enumerate([n] + list(range(n))):
    #     axes[0, ii].imshow(imgs_src[i])
    #     axes[1, ii].imshow(imgs_tar[i])
    #     axes[0, ii].axis('off')
    #     axes[1, ii].axis('off')
    # plt.show()


def all_to_new():
    np.random.seed(0)
    shape_indices = np.random.choice(4745, 5)  # choose between [0, 4745]
    n = len(shape_indices)
    imgs_src = [None for _ in range(n+1)]
    imgs_tar = [None for _ in range(n+1)]
    faces = [None for _ in range(n+1)]
    verts = [None for _ in range(n+1)]  # verts tensor
    mesh_o = [None for _ in range(n+1)]  # mesh original
    mesh_d = [None for _ in range(n+1)]  # mesh deformed
    l_new = torch.from_numpy(lat_new).to(device)

    for i, idx in enumerate(shape_indices):
        v, f = dset.get_single(idx)
        verts[i] = v[None].to(device)
        faces[i] = f.detach().cpu().numpy()
    verts[-1] = torch.from_numpy(mesh_new.vertices)[None].float().to(device)
    faces[-1] = mesh_new.faces

    lat_path = lambda l_src_, l_tar_: torch.stack([l_src_, torch.zeros_like(l_src_), l_tar_], dim=1)
    ten2npy = lambda tensor: tensor.detach().cpu().numpy()

    pbar = tqdm(total=n+1)

    with torch.no_grad():
        for i, ii in enumerate(shape_indices):
            # get the latent codes corresponding to these shapes
            l_src = deformer.get_lat_params(ii)[None] 
            l_tar = l_new  # target is the new shape

            # deform source to target
            v_s2t = deformer(verts[i], lat_path(l_src, l_tar))[0]  # source to target
            mesh_d[i] = trimesh.Trimesh(ten2npy(v_s2t), faces[i], process=False)
            mesh_o[i] = trimesh.Trimesh(ten2npy(verts[i][0]), faces[i], process=False)
            pbar.update(1)
            
        # new observation
        l_src = torch.from_numpy(lat_new).to(device)
        l_tar = l_new
        # deform source to target
        v_s2t = deformer(verts[-1], lat_path(l_src, l_tar))[0]  # source to target
        mesh_d[-1] = trimesh.Trimesh(ten2npy(v_s2t), faces[-1], process=False)
        mesh_o[-1] = trimesh.Trimesh(ten2npy(verts[-1][0]), faces[-1], process=False)
        pbar.update(1)
            
    pbar.close()

    for idx, (mo, md) in enumerate(zip(mesh_o, mesh_d)):
        color = (md.vertices.copy() - bboxes_d[0]) / (bboxes_d[1] - bboxes_d[0])
        md_ = md.copy()#; md_.vertices *= 0.5
        mo_ = mo.copy()
        img_tar, _, _, _ = render.render_trimesh(md_, eye_1, center, up, res=(224, 224))
        img_src, _, _, _ = render.render_trimesh(mo_, eye_1, center, up, res=(224, 224))
        
        imgs_src[idx] = img_src
        imgs_tar[idx] = img_tar
        
    # plot transformation table
    fig, axes = plt.subplots(figsize=(5*(n+1), 10), nrows=2, ncols=(n+1))
    for i in range(n+1):
        axes[0, i].imshow(imgs_src[i])
        axes[1, i].imshow(imgs_tar[i])
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()


new_observation()
