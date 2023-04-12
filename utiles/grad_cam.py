from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
# from torchvision.models import resnet50,resnet101
from PIL import Image
import cv2
import numpy as np
import os
# import torchsummary
import torch

# import utiles.log as log
import sys
sys.path.append('/mnt/llz/code/cls/mymodel/')
from models.pretrainedModel.pretrainmodels import get_resnet101,Net,get_convnext_base

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def func(model,img_path,cam_path):
    # rgb_img = cv2.imread(img_path, 1)  # imread()读取的是BGR格式
    # rgb_img=rgb_img.resize((224,224))
    img=Image.open(img_path)
    # img=img.resize((224,224))
    img=img.convert('RGB')
    rgb_img = np.array(img, dtype=np.float32)
    rgb_img = np.float32(rgb_img) / 255
    # rgb_img=rgb_img.squeeze()
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    print("input tensor shape: ",input_tensor.shape)
    # target_layer = [model.layer4[-1]]
    target_layer=[model.classification_head1[1]]
    print("target_layer : ",target_layer)
    grad_cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
    eighten_cam = EigenCAM(model=model, target_layers=target_layer, use_cuda=False)
    score_cam= ScoreCAM(model=model, target_layers=target_layer, use_cuda=False)
    grad_campp=GradCAMPlusPlus(model=model, target_layers=target_layer, use_cuda=False)
    ab_cam=AblationCAM(model=model, target_layers=target_layer, use_cuda=False)
    xgrad_cam=XGradCAM(model=model, target_layers=target_layer, use_cuda=False)

    target_category=2
    # grayscale_gradcam = grad_cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    # grayscale_eightencam = eighten_cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    # grayscale_scorecam = score_cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    # grayscale_campp = grad_campp(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    # grayscale_abcam = ab_cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]
    # grayscale_xgradcam = xgrad_cam(input_tensor=input_tensor, target_category=target_category)  # [batch, 224,224]

    # ----------------------------------
    '''
    6)展示热力图并保存
    '''
    # In this example grayscale_cam has only one image in the batch:
    # 7.展示热力图并保存, grayscale_cam是一个batch的结果，只能选择一张进行展示
    # grayscale_campp = grayscale_campp[0]
    # grayscale_gradcam = grayscale_gradcam[0]
    # grayscale_abcam=grayscale_abcam[0]
    # grayscale_eightencam=grayscale_eightencam[0]
    # grayscale_scorecam=grayscale_scorecam[0]
    # grayscale_xgradcam=grayscale_xgradcam[0]
    # cams=[grayscale_campp,grayscale_gradcam,grayscale_abcam,grayscale_eightencam,grayscale_scorecam,grayscale_xgradcam]
    cams = [grad_cam, grad_campp, ab_cam, score_cam, xgrad_cam, eighten_cam]
    if not os.path.exists(os.path.join(cam_path,'cam_rst')):
        os.mkdir(os.path.join(cam_path,'cam_rst'))
    for cam in cams:
        # grayscale_cam=cam(input_tensor=input_tensor, target_category=target_category)
        grayscale_cam = cam(input_tensor=input_tensor,eigen_smooth=True,aug_smooth=True)
        grayscale_cam=grayscale_cam[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam)  # (224, 224, 3)

        cv2.imwrite(os.path.join(cam_path,'cam_rst','{}_first_try.jpg'.format(cam.__class__.__name__)), visualization)
        # cv2.imwrite(os.path.join('.', '{}_first_try.jpg'.format(cam.__class__.__name__)),
        #             visualization)


def load_model(model_path):
    print("torch vision: ",torch.__version__)
    model,target = get_resnet101()
    net=Net(model,target).cpu()
    net_pth = torch.load(
        model_path,map_location=torch.device('cpu'))
    net.load_state_dict(net_pth)
    return net

if __name__ == '__main__':
    model_path="/mnt/llz/log/2022-08-17/save_models/fold_0_epoch_21.pt"
    cam_path='/mnt/llz/log/pred'
    model = load_model(model_path)  # 预先训练
    for name in model.named_modules():
        print(name)
    print("model loaded!")
    image_path = '/mnt/llz/media/myNpcDiagnoseProjectDataset/test/npc/JD0001_05MEQSQM_39.png'
    func(model,image_path,cam_path)
