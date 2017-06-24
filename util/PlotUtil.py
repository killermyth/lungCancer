#!/usr/bin/env python
# encoding: utf-8
# coding:utf-8

"""
工具类
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure, morphology


class PlotUtil(object):
    @staticmethod
    def show_lung_npy(img_name, mask_name):
        imgs = np.load(img_name)
        masks = np.load(mask_name)
        for i in range(100, len(imgs)):
            print ("图片的第 %d 层" % i)
            fig, ax = plt.subplots(2, 2, figsize=[8, 8])
            ax[0, 0].imshow(imgs[i])
            ax[0, 0].set_title(u'彩色切片')
            ax[0, 1].imshow(imgs[i], cmap='gray')
            ax[0, 1].set_title(u'黑白切片')
            ax[1, 0].imshow(masks[i], cmap='gray')
            ax[1, 0].set_title(u'肺部')
            ax[1, 1].imshow(imgs[i] * masks[i], cmap='gray')
            ax[1, 1].set_title(u'肺部切片')
            plt.show()
            print ('\n\n')

    @staticmethod
    def show_node_npy(img_name, mask_name, node_name):
        imgs = np.load(img_name)
        masks = np.load(mask_name)
        nodes = np.load(node_name)
        for i in range(100, len(imgs)):
            print ("the level %d of img" % i)
            fig, ax = plt.subplots(3, 2, figsize=[8, 8])
            ax[0, 0].imshow(imgs[i])
            ax[0, 0].set_title(u'color slice')
            ax[0, 1].imshow(imgs[i], cmap='gray')
            ax[0, 1].set_title(u'gray slice')
            ax[1, 0].imshow(masks[i], cmap='gray')
            ax[1, 0].set_title(u'lung')
            ax[1, 1].imshow(imgs[i] * masks[i], cmap='gray')
            ax[1, 1].set_title(u'lung slice')
            ax[2, 0].imshow(nodes[i], cmap='gray')
            ax[2, 0].set_title(u'node')
            ax[2, 1].imshow(imgs[i] * nodes[i], cmap='gray')
            ax[2, 1].set_title(u'node slice')
            plt.show()
            print ('\n\n')

    @staticmethod
    def plot_3d(image, threshold=-300):
        # Position the scan upright,
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2, 1, 0)

        verts, faces = measure.marching_cubes(p, threshold)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

        plt.show()
