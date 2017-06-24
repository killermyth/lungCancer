#!/usr/bin/env python
# encoding: utf-8

"""
从单个mhd文件中切割出对应的结节
转换为npy待训练使用
"""

import SimpleITK as sitk
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

output_path = "./train_npy/"


class TrainNodCrop(object):
    def __init__(self, file_path, df_nodes):
        self.file_path = file_path
        self.df_nodes = df_nodes

    def crop(self):
        itk_img = sitk.ReadImage(self.file_path)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        # print self.df_nodes
        for node_idx, cur_row in self.df_nodes.iterrows():
            node_x = cur_row['coordX']
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            # just keep 3 slices
            imgs = np.ndarray([3, height, width], dtype=np.float32)
            masks = np.ndarray([3, height, width], dtype=np.uint8)
            center = np.array([node_x, node_y, node_z], dtype=np.float64)  # nodule center
            v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
            # clip prevents going out of bounds in Z
            for i, i_z in enumerate(np.arange(int(v_center[2]) - 1, int(v_center[2]) + 2).clip(0, num_z - 1)):
                mask = self.make_mask(center, diam, i_z * spacing[2] + origin[2], width, height, spacing, origin)
                masks[i] = mask
                imgs[i] = img_array[i_z]
            print cur_row['seriesuid']
            np.save(os.path.join(output_path, "images_%04d_%s.npy" % (node_idx, cur_row['seriesuid'])), imgs)
            np.save(os.path.join(output_path, "masks_%04d_%s.npy" % (node_idx, cur_row['seriesuid'])), masks)

    def make_mask(self, center, diam, z, width, height, spacing, origin):
        '''
            Center : 圆的中心 px -- list of coordinates x,y,z
            diam : 圆的直径 px -- diameter
            widthXheight : pixel dim of image
            spacing = mm/px conversion rate np array x,y,z
            origin = x,y,z mm np.array
            z = z position of slice in world coordinates mm
        '''
        mask = np.zeros([height, width])  # 0's everywhere except nodule swapping x,y to match img
        # convert to nodule space from world coordinates

        # Defining the voxel range in which the nodule falls
        v_center = (center - origin) / spacing
        v_diam = int(float(diam) / spacing[0] + 5)
        v_xmin = np.max([0, int(v_center[0] - v_diam) - 5])
        v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + 5])
        v_ymin = np.max([0, int(v_center[1] - v_diam) - 5])
        v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + 5])

        v_xrange = range(v_xmin, v_xmax + 1)
        v_yrange = range(v_ymin, v_ymax + 1)

        # Convert back to world coordinates for distance calculation
        x_data = [x * spacing[0] + origin[0] for x in range(width)]
        y_data = [x * spacing[1] + origin[1] for x in range(height)]

        # Fill in 1 within sphere around nodule
        for v_x in v_xrange:
            for v_y in v_yrange:
                p_x = spacing[0] * v_x + origin[0]
                p_y = spacing[1] * v_y + origin[1]
                if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                    mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
        return mask

    def show_npy(self, img_name, mask_name):
        imgs = np.load(img_name)
        masks = np.load(mask_name)
        for i in range(len(imgs)):
            print ("图片的第 %d 层" % i)
            fig, ax = plt.subplots(2, 2, figsize=[8, 8])
            ax[0, 0].imshow(imgs[i])
            ax[0, 0].set_title(u'彩色切片')
            ax[0, 1].imshow(imgs[i], cmap='gray')
            ax[0, 1].set_title(u'黑白切片')
            ax[1, 0].imshow(masks[i], cmap='gray')
            ax[1, 0].set_title(u'节点')
            ax[1, 1].imshow(imgs[i] * masks[i], cmap='gray')
            ax[1, 1].set_title(u'节点切片')
            plt.show()
            print ('\n\n')


if __name__ == '__main__':
    file_name = './LKDS-00001.mhd'
    node = {'coordX': '-76.4498793983', 'coordY': '-49.5405710363', 'coordZ': '229.5', 'diameter_mm': '14.1804045239'}
    nodes = [node]
    nc = TrainNodCrop(file_name, nodes)
    # nc.crop()
    image_name = './images_0001_1.npy'
    mask_name = './masks_0001_1.npy'

    nc.show_npy(image_name, mask_name)
