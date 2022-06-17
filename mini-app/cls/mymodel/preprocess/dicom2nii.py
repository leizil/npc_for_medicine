import os
import pandas
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def dicom2nii(dicom_path, nii_path='..', n=-1, mode='nii.gz'):
    """

    :param dicom_path: 有很多.dcm文件的上级目录
    :param nii_path: 存放的路径
    :param n: 为了批量转换用的
    :param mode: 转换为nii.gz 或者nii
    :return:
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    print(dicom_names[0])
    id = dicom_names[0].split('/')[-1]
    t = id.split('_')[0]
    p = id.split('_')[1]

    image = reader.Execute()
    if mode == 'nii':
        if n!=-1:
            sitk.WriteImage(image, os.path.join(nii_path, t + '_' + p + '_' + str(n) + '.nii'))
        else:
            sitk.WriteImage(image, os.path.join(nii_path, t + '_' + p + '.nii'))
        print(os.path.join(nii_path, id + '.nii'), '成功')
    elif mode == 'nii.gz':
        if n!=-1:
            sitk.WriteImage(image, os.path.join(nii_path, t + '_' + p + '_' + str(n) + '.nii.gz'))
        else:
            sitk.WriteImage(image, os.path.join(nii_path, t + '_' + p + '.nii.gz'))
        print(os.path.join(nii_path, id + '.nii.gz'), '成功')
    else:
        print('unknown mode,mode is name.nii / name.nii.gz')
        return -1

if __name__ == '__main__':
    dicom_path=''
    dicom2nii(dicom_path)


