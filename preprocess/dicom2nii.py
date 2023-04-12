import os
import pandas
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def dicom2nii(dicom_path, nii_path='..', n=""):
    """

    :param dicom_path: 有很多.dcm文件的上级目录
    :param nii_path: 存放的路径
    :return:
    """

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
    reader.SetFileNames(dicom_names)
    # print("dicom_names[0]",dicom_names[0])
    model = dicom_names[0].split('/')[-2]
    t = model.split('_')[0]
    # p = id.split('_')[1]
    # print("n: ",n)
    # print("model : ", model)
    # print("t: ", t)
    image = reader.Execute()

    sitk.WriteImage(image, os.path.join(nii_path,"PYBEFORE2017_"+ model+"_"+n + '.nii.gz'))

    print(os.path.join(nii_path,"PYBEFORE2018_"+ model+"_"+n + '.nii.gz'), '成功')



def test():
    dicom_path = '/mnt/llz/dataset/npc/dicom/PY_BEFORE2018'
    nii_path = "/mnt/llz/dataset/npc/temp/nii"
    n_s = os.listdir(dicom_path)
    for n in n_s:
        print("  n:", n)
        n_path = os.path.join(dicom_path, n)
        time = os.listdir(n_path)
        # print(time)
        for t in time:
            t_path = os.path.join(n_path, t)
            mri_names = os.listdir(t_path)
            # print(mri_names)
            for mri_name in mri_names:
                mri_path = os.path.join(t_path, mri_name)
                # nii_save_path=os.path.join(nii_path,)
                dicom2nii(mri_path, nii_path, n)
                # return 0

if __name__ == '__main__':
    test()





