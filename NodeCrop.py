#!/usr/bin/env python
# encoding: utf-8

"""
从单个mhd文件中切割出对应的肺部
转换为npy待训练使用
"""

from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, \
    reconstruction, binary_closing
from skimage.measure import label, regionprops, perimeter
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import scipy
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
import os
import pandas as pd
from util.PlotUtil import PlotUtil


class NodeCrop(object):
    def __init__(self, img_path, output_path, df_nodes):
        self.img_path = img_path
        array = self.img_path.split('/')
        self.img_id = array[len(array) - 1].split('.')[0]
        self.output_path = output_path
        self.df_nodes = df_nodes

    def load_itk(self):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(self.img_path)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)

        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ct_scan, origin, spacing

    """
    世界坐标转换为voxel coordinates
    """

    def world_2_voxel(self, world_coordinates, origin, spacing):
        stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
        voxel_coordinates = stretched_voxel_coordinates / spacing
        return voxel_coordinates

    """
    voxel coordinates转换为世界坐标
    """

    def voxel_2_world(self, voxel_coordinates, origin, spacing):
        stretched_voxel_coordinates = voxel_coordinates * spacing
        world_coordinates = stretched_voxel_coordinates + origin
        return world_coordinates

    # 输入mhd文件，rescaling to 1mm size in all directions之后切割出肺部以及结节并转为npy
    def create_lung_nodule_mask(self):
        # if os.path.isfile(imagePath.replace('original',SAVE_FOLDER_image)) == False:
        img, origin, spacing = self.load_itk()

        # calculate resize factor
        RESIZE_SPACING = [1, 1, 1]
        resize_factor = spacing / RESIZE_SPACING
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img.shape
        new_spacing = spacing / real_resize

        # resize image
        lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)

        # Segment the lung structure
        lung_img = lung_img + 1024
        lung_mask = self.segment_lung_from_ct_scan(lung_img)
        lung_img = lung_img - 1024

        # create nodule mask
        nodule_mask = self.draw_circles(lung_img, origin, new_spacing)

        lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], 512, 512)), np.zeros(
            (lung_mask.shape[0], 512, 512)), np.zeros((nodule_mask.shape[0], 512, 512))

        original_shape = lung_img.shape
        for z in range(lung_img.shape[0]):
            offset = (512 - original_shape[1])
            upper_offset = np.round(offset / 2)
            lower_offset = offset - upper_offset

            new_origin = self.voxel_2_world([-upper_offset, -lower_offset, 0], origin, new_spacing)

            lung_img_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_img[z, :, :]
            lung_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = lung_mask[z, :, :]
            nodule_mask_512[z, upper_offset:-lower_offset, upper_offset:-lower_offset] = nodule_mask[z, :, :]

        # save images.
        np.save(os.path.join(self.output_path, "lung_img_%s.npy" % self.img_id), lung_img_512)
        np.save(os.path.join(self.output_path, "lung_mask_%s.npy" % self.img_id), lung_mask_512)
        np.save(os.path.join(self.output_path, "nodule_mask_%s.npy" % self.img_id), nodule_mask_512)

    def draw_circles(self, image, origin, spacing):
        # make empty matrix, which will be filled with the mask
        RESIZE_SPACING = [1, 1, 1]
        image_mask = np.zeros(image.shape)

        # run over all the nodules in the lungs
        for node_idx, cur_row in self.df_nodes.iterrows():
            coord_x = cur_row['coordX']
            coord_y = cur_row["coordY"]
            coord_z = cur_row["coordZ"]
            radius = cur_row["diameter_mm"] / 2
            image_coord = np.array((coord_z, coord_y, coord_x))

            # determine voxel coordinate given the worldcoordinate
            image_coord = self.world_2_voxel(image_coord, origin, spacing)

            # determine the range of the nodule
            noduleRange = self.seq(-radius, radius, RESIZE_SPACING[0])

            # create the mask
            for x in noduleRange:
                for y in noduleRange:
                    for z in noduleRange:
                        coords = self.world_2_voxel(np.array((coord_z + z, coord_y + y, coord_x + x)), origin, spacing)
                        if (np.linalg.norm(image_coord - coords) * RESIZE_SPACING[0]) < radius:
                            image_mask[np.round(coords[0]), np.round(coords[1]), np.round(coords[2])] = int(1)

        return image_mask

    def seq(self, start, stop, step=1):
        n = int(round((stop - start) / float(step)))
        if n > 1:
            return ([start + step * i for i in range(n + 1)])
        else:
            return ([])

    def segment_lung_from_ct_scan(self, ct_scan):
        return np.asarray([self.get_segmented_lungs(slice) for slice in ct_scan])

    def get_segmented_lungs(self, im, plot=False):
        '''
        This funtion segments the lungs from the given 2D slice.
        '''
        if plot == True:
            f, plots = plt.subplots(8, 1, figsize=(5, 40))
        '''
        Step 1: Convert into a binary image. 
        '''
        binary = im < 604
        if plot == True:
            plots[0].axis('off')
            plots[0].imshow(binary, cmap=plt.cm.bone)
        '''
        Step 2: Remove the blobs connected to the border of the image.
        '''
        cleared = clear_border(binary)
        if plot == True:
            plots[1].axis('off')
            plots[1].imshow(cleared, cmap=plt.cm.bone)
        '''
        Step 3: Label the image.
        '''
        label_image = label(cleared)
        if plot == True:
            plots[2].axis('off')
            plots[2].imshow(label_image, cmap=plt.cm.bone)
        '''
        Step 4: Keep the labels with 2 largest areas.
        '''
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                        label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0
        if plot == True:
            plots[3].axis('off')
            plots[3].imshow(binary, cmap=plt.cm.bone)
        '''
        Step 5: Erosion operation with a disk of radius 2. This operation is 
        seperate the lung nodules attached to the blood vessels.
        '''
        selem = disk(2)
        binary = binary_erosion(binary, selem)
        if plot == True:
            plots[4].axis('off')
            plots[4].imshow(binary, cmap=plt.cm.bone)
        '''
        Step 6: Closure operation with a disk of radius 10. This operation is 
        to keep nodules attached to the lung wall.
        '''
        selem = disk(10)
        binary = binary_closing(binary, selem)
        if plot == True:
            plots[5].axis('off')
            plots[5].imshow(binary, cmap=plt.cm.bone)
        '''
        Step 7: Fill in the small holes inside the binary mask of lungs.
        '''
        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)
        if plot == True:
            plots[6].axis('off')
            plots[6].imshow(binary, cmap=plt.cm.bone)
        '''
        Step 8: Superimpose the binary mask on the input image.
        '''
        get_high_vals = binary == 0
        im[get_high_vals] = 0
        if plot == True:
            plots[7].axis('off')
            plots[7].imshow(im, cmap=plt.cm.bone)

        return im


if __name__ == '__main__':
    file_path = './LKDS-00001.mhd'
    train_output_path = "./train_npy/"
    d = {'index': 65, 'seriesuid': 'LKDS-00001', 'coordX': -76.449879, 'coordY': -49.540571, 'coordZ': 229.5,
         'diameter_mm': 14.180405}
    df = pd.DataFrame(data=d, index=['index'])
    # nodeCrop = NodeCrop(file_path, train_output_path, df)
    # nodeCrop.create_lung_nodule_mask()
    PlotUtil.show_node_npy(train_output_path + "lung_img_LKDS-00001.npy",
                           train_output_path + "lung_mask_LKDS-00001.npy",
                           train_output_path + "nodule_mask_LKDS-00001.npy")
