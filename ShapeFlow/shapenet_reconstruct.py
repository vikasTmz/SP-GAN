"""Reconstruct shape from point cloud using learned deformation space.
"""
import os
import sys
import argparse
import json
import trimesh
import torch
import numpy as np
import time
from types import SimpleNamespace

from shapenet_dataloader import ShapeNetMesh, FixedPointsCachedDataset
from shapeflow.layers.deformation_layer import NeuralFlowDeformer
from shapenet_embedding import LatentEmbedder

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces = []
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()
            self.vertices = np.array(self.vertices)
        except IOError:
            print(".obj file not found.")

class PlyLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.colors = []
        ##
        try:
            f = open(fileName)
            i = 0
            for line in f:
                if i > 12:
                    splits = line.split(" ")
                    splits = list(filter(None, splits))
                    vertex = (float(splits[0]), float(splits[1]), float(splits[2]))
                    self.vertices.append(vertex)
                    color = (float(splits[3]), float(splits[4]), float(splits[5]))
                    self.colors.append(color)
                i+=1
            f.close()
            self.vertices = np.array(self.vertices)
            self.colors = np.array(self.colors)
        except IOError:
            print(".obj file not found.")

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

synset_to_cat = {
    "02691156": "airplane",
    "02933112": "cabinet",
    "03001627": "chair",
    "03636649": "lamp",
    "04090263": "rifle",
    "04379243": "table",
    "04530566": "watercraft",
    "02828884": "bench",
    "02958343": "car",
    "03211117": "display",
    "03691459": "speaker",
    "04256520": "sofa",
    "04401088": "telephone",
}

cat_to_synset = {value: key for key, value in synset_to_cat.items()}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reconstructions via retrieve and deform."
    )

    parser.add_argument(
        "--input_path_1",
        type=str,
        default="3dfuture_5_pc.ply",
        # default="/media/andy/Elements/Shapeflow_data/data/shapenet_simplified/val/03001627/c4f9249def12870a2b3e9b6eb52d35df/model.ply",
        # default="/media/andy/Elements/Shapeflow_data/data/shapenet_simplified/val/03001627/bcc73b8ff332b4df3d25ee35360a1f4d/model.ply",
        help="path to input points (.ply file).",
    )
    parser.add_argument(
        "--input_path_2",
        type=str,
        default="3dfuture_6_pc.ply",
        help="path to input points (.ply file).",
    )

    parser.add_argument(
        "--output_dir", type=str, default="", help="path to output meshes."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="top k nearest neighbor to retrieve.",
    )
    parser.add_argument(
        "-ne",
        "--embedding_niter",
        type=int,
        default=30,
        help="number of embedding iterations.",
    )
    parser.add_argument(
        "-nf",
        "--finetune_niter",
        type=int,
        default=30,
        help="number of finetuning iterations.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/media/andy/Elements/Shapeflow_data/runs/pretrained_ckpt",
        help="path to pretrained checkpoint "
             "(params.json must be in the same directory).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to run inference on.",
    )
    args = parser.parse_args()
    return args


def main():
    t0 = time.time()
    args_eval = get_args()

    device = torch.device(args_eval.device)

    # load training args
    run_dir = args_eval.checkpoint
    print(run_dir)
    args = SimpleNamespace(
        **json.load(open(os.path.join(run_dir, "params.json"), "r"))
    )

    # assert category is correct
    mesh_name = args_eval.input_path_1.split("/")[-1]

    # output directories
    # mesh_out_dir = os.path.join(args_eval.output_dir, "meshes", syn_id)

    meta_out_dir = os.path.join(
        args_eval.output_dir, "meta", mesh_name.replace(".ply", "")
    )
    orig_dir = os.path.join(meta_out_dir, "original_retrieved")
    deformed_dir = os.path.join(meta_out_dir, "deformed")

    # initialize deformer
    # input points
    sample_points = 2048
    # mesh_1 = trimesh.load(args_eval.input_path_1)
    # mesh_2 = trimesh.load(args_eval.input_path_2)

    mesh_1 = PlyLoader(args_eval.input_path_1)
    mesh_2 = PlyLoader(args_eval.input_path_2)

    # mesh_v = np.array(mesh_1.vertices)
    # points_1 = mesh_1.sample(sample_points)
    # points_2 = mesh_2.sample(sample_points)
    # indices = np.random.randint(mesh_1.vertices.shape[0], size=sample_points)
    points_1 = mesh_1.vertices[:sample_points]
    
    # indices = np.random.randint(mesh_2.vertices.shape[0], size=sample_points)
    points_2 = mesh_2.vertices[:sample_points]

    # points = mesh_v[:1024]
    # export_obj_cpu('shapenet_recon_input.obj', points, random_trans=[-1.5,0,0])
    # dataloader
    data_root = args.data_root
    mesh_dataset = ShapeNetMesh(
        data_root=data_root,
        split="train",
        category="chair",
        normals=False,
    )
    point_dataset = FixedPointsCachedDataset(
        "/media/andy/Elements/Shapeflow_data/data/shapenet_pointcloud/train/03001627.pkl",
        npts=sample_points,
    )

    # setup model
    deformer = NeuralFlowDeformer(
        latent_size=args.lat_dims,
        f_width=args.deformer_nf,
        s_nlayers=2,
        s_width=5,
        method=args.solver,
        nonlinearity=args.nonlin,
        arch="imnet",
        adjoint=args.adjoint,
        rtol=args.rtol,
        atol=args.atol,
        via_hub=True,
        no_sign_net=(not args.sign_net),
        symm_dim=(2 if args.symm else None),
    )

    lat_params = torch.nn.Parameter(
        torch.randn(mesh_dataset.n_shapes, args.lat_dims) * 1e-1,
        requires_grad=True,
    )
    deformer.add_lat_params(lat_params)
    deformer.to(device)

    # load checkpoint
    resume_dict = torch.load(os.path.join(args_eval.checkpoint,"checkpoint_latest.pth.tar_deepdeform_100.pth.tar"))
    deformer.load_state_dict(resume_dict["deformer_state_dict"])

    # embed
    embedder = LatentEmbedder(point_dataset, mesh_dataset, deformer, topk=5)

    # Embed shape 1
    lat_codes_pre_1, lat_codes_post_1 = embedder.embed(
        torch.tensor(points_1)[None].to(device),
        matching="two_way",
        verbose=True,
        lr=1e-2,
        embedding_niter=args_eval.embedding_niter,
        finetune_niter=args_eval.finetune_niter,
        bs=4,
        seed=1,
    )

    # Embed shape 2
    lat_codes_pre_2, lat_codes_post_2 = embedder.embed(
        torch.tensor(points_2)[None].to(device),
        matching="two_way",
        verbose=True,
        lr=1e-2,
        embedding_niter=args_eval.embedding_niter,
        finetune_niter=args_eval.finetune_niter,
        bs=4,
        seed=1,
    )

    print("Done embedding...")

    # retrieve deformed models
    embedder.dense_correspondence(
        lat_codes_pre_1,
        lat_codes_pre_2,
        torch.tensor(points_1)[None].to(device),
        torch.tensor(points_2)[None].to(device),
        torch.tensor(mesh_1.colors[:sample_points])[None].to(device),
        torch.tensor(mesh_2.colors[:sample_points])[None].to(device),
        "pre"
    )

    # retrieve deformed models
    embedder.dense_correspondence(
        lat_codes_post_1,
        lat_codes_post_2,
        torch.tensor(points_1)[None].to(device),
        torch.tensor(points_2)[None].to(device),
        torch.tensor(mesh_1.colors[:sample_points])[None].to(device),
        torch.tensor(mesh_2.colors[:sample_points])[None].to(device),
        "post"
    )

    print("Done deforming new shapes...")
    exit()

    # retrieve deformed models
    deformed_meshes, orig_meshes, dist = embedder.retrieve(
        lat_codes_post, tar_pts=points, matching="two_way"
    )
    print("Done retrieving...")

    asort = np.argsort(dist)
    dist = [dist[i] for i in asort]
    deformed_meshes = [deformed_meshes[i] for i in asort]
    orig_meshes = [orig_meshes[i] for i in asort]

    # output best mehs
    vb, fb = deformed_meshes[0]

    # trimesh.Trimesh(vb, fb).export(mesh_out_file)
    export_obj_cpu("shapenet_recon_meshoutfile.obj",vb,random_trans=[0,0,0])

    # meta directory
    for i in range(len(deformed_meshes)):
        vo, fo = orig_meshes[i]
        vd, fd = deformed_meshes[i]
        export_obj_cpu("shapenet_recon_orig_%d.obj"%(i),vo,random_trans=[i*1.5,1.5,0])
        export_obj_cpu("shapenet_recon_deformed_%d.obj"%(i),vd,random_trans=[i*1.5,3,0])
        # trimesh.Trimesh(vo, fo).export(os.path.join(orig_dir, f"{i}.ply"))
        # trimesh.Trimesh(vd, fd).export(os.path.join(deformed_dir, f"{i}.ply"))
    # np.save(os.path.join(meta_out_dir, "latent.npy"), lat_codes_pre)
    t1 = time.time()
    print(f"Total Timelapse: {t1-t0:.4f}")


if __name__ == "__main__":
    main()
