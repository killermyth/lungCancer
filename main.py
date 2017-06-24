# -*- coding: utf-8 -*-
# !/usr/bin/env python

import SimpleITK as sitk
import cv2
from FileWorker import FileWorker
from pandas import DataFrame

train_data_path = "../train_subset00/"
mhd_suffix = ".mhd"
zraw_suffix = ".zraw"
file_name = "LKDS-00001"


def train_crop():
    worker = FileWorker()
    worker.train_start()


def lung_crop():
    worker = FileWorker()
    worker.test_start()


def main():
    lung_crop()


if __name__ == '__main__':
    main()
