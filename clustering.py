import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import argparse
import sys
from scene import GaussianModel
from gaussian_renderer import render
import cv2
from arguments import PipelineParams
import numpy as np
from plyfile import PlyData, PlyElement
import os

class DinoModel():
    def load_model(self, model_path):
        print("\n[Loading the gaussian model]")

        saved_dict = torch.load(model_path)

        self.active_sh_degree = saved_dict['active_sh_degree']
        self.max_sh_degree = saved_dict['max_sh_degree']
        self.xyz = saved_dict['xyz']
        self.features_dc = saved_dict['features_dc']
        self.features_rest = saved_dict['features_rest']
        self.scaling = saved_dict['scaling']
        self.rotation = saved_dict['rotation']
        self.opacity = saved_dict['opacity']
        self.max_radii2D = saved_dict['max_radii2D']
        self.xyz_gradient_accum = saved_dict['xyz_gradient_accum']
        self.denom = saved_dict['denom']
        self.dino_feat1 = saved_dict['dino_feat1']
        self.dino_feat2 = saved_dict['dino_feat2']
        self.dino_feat3 = saved_dict['dino_feat3']
        self.dino_feat4 = saved_dict['dino_feat4']
        self.cameras = saved_dict['cameras']

        self.gaussians = GaussianModel(0)

        self.gaussians.active_sh_degree = self.active_sh_degree
        self.gaussians.max_sh_degree = self.max_sh_degree
        self.gaussians._xyz = nn.Parameter(self.xyz)
        self.gaussians._features_dc = nn.Parameter(self.features_dc)
        self.gaussians._features_rest = nn.Parameter(self.features_rest)
        self.gaussians._dino_feat1 = nn.Parameter(self.dino_feat1)
        self.gaussians._dino_feat2 = nn.Parameter(self.dino_feat2)
        self.gaussians._dino_feat3 = nn.Parameter(self.dino_feat3)
        self.gaussians._dino_feat4 = nn.Parameter(self.dino_feat4)
        self.gaussians._dino_feats = [self.gaussians._dino_feat1, self.gaussians._dino_feat2, self.gaussians._dino_feat3, self.gaussians._dino_feat4]
        self.gaussians._scaling = nn.Parameter(self.scaling)
        self.gaussians._rotation = nn.Parameter(self.rotation)
        self.gaussians._opacity = nn.Parameter(self.opacity)

        print("[Gaussian model created successfully]")

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.gaussians._features_dc.shape[1]*self.gaussians._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.gaussians._features_rest.shape[1]*self.gaussians._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.gaussians._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.gaussians._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_model(self, save_path):
        xyz = self.gaussians._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.gaussians._opacity.detach().cpu().numpy()
        scale = self.gaussians._scaling.detach().cpu().numpy()
        rotation = self.gaussians._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(save_path, 'point_cloud.ply'))

    def __init__(self, args, pipe):
        self.load_model(args.model_path)
        self.pipe = pipe

    def render(self, cam_id, save=False):
        print("\n[Rendering: {}]".format(cam_id))

        viewpoint_camera = self.cameras[cam_id]

        bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")

        render_pkg = render(viewpoint_camera, self.gaussians, self.pipe, bg, mode='rgb')
        image = render_pkg['render']

        image = np.array(image.permute(1, 2, 0).cpu().detach().numpy())

        # save the image
        if save == True:
            cv2.imwrite('rendered_image.png', image)

        print("[Rendering done]")

    def cluster(self, k):
        # Clustering
        print("\n[Clustering]")

        # contatenate the dino feats for clustering
        print(self.gaussians._dino_feats[0].shape)
        dino_feats = torch.cat(self.gaussians._dino_feats, dim=1)
        print(dino_feats.shape)
        dino_feats = dino_feats.reshape(-1, dino_feats.shape[1])
        print(dino_feats.shape)

        # cluster the dino feats with faiss gpu kmeans
        ncentroids = k
        niter = 20
        verbose = True
        d = dino_feats.shape[1]
        dinofeats_np = dino_feats.cpu().detach().numpy()
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
        kmeans.train(dinofeats_np)

        # get the cluster centers
        centroids = kmeans.centroids

        # getting the cluster assignments
        distances, indices = kmeans.index.search(dinofeats_np, 1)

        # saving each cluster to different ply file
        for i in range(ncentroids):
            print("Cluster: ", i)
            cluster_indices = np.where(indices == i)[0]
            xyz = self.gaussians._xyz.detach().cpu().numpy()
            normals = np.zeros_like(xyz)
            f_dc = self.gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            f_rest = self.gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opacities = self.gaussians._opacity.detach().cpu().numpy()
            scale = self.gaussians._scaling.detach().cpu().numpy()
            rotation = self.gaussians._rotation.detach().cpu().numpy()

            cluster_xyz = xyz[cluster_indices]
            cluster_normals = normals[cluster_indices]
            cluster_f_dc = f_dc[cluster_indices]
            cluster_f_rest = f_rest[cluster_indices]
            cluster_opacities = opacities[cluster_indices]
            cluster_scale = scale[cluster_indices]
            cluster_rotation = rotation[cluster_indices]

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

            elements = np.empty(cluster_xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((cluster_xyz, cluster_normals, cluster_f_dc, cluster_f_rest, cluster_opacities, cluster_scale, cluster_rotation), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write('cluster_{}.ply'.format(i))

if __name__ == "__main__":
    #Set up command line argument parser
    parser = argparse.ArgumentParser(description='Clustering gaussians with dino features')
    pp = PipelineParams(parser)
    parser.add_argument('--model_path', type=str, help='Path to the model folder')
    parser.add_argument('--save_path', type=str, help='Path to save the clustering results', default='.')
    parser.add_argument('--k', type=int, help='Number of clusters', default=20)
    parser.add_argument('--d', type=int, help='Number of dimensions', default=12)
    args = parser.parse_args(sys.argv[1:])

    print("Gaussian model: " + args.model_path)

    # loading the gaussian model
    scene = DinoModel(args, pp.extract(args))

    # # render an image for testing
    # scene.render(1, save=True)

    scene.cluster(args.k)

    scene.save_model(args.save_path)
