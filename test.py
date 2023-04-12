import sys
import torch
import pandas as pd
import numpy as np
import torchsummary
import tqdm
from PIL import Image
from torchvision.transforms import CenterCrop
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from torchcam.methods import GradCAMpp,SmoothGradCAMpp
from torchcam.utils import overlay_mask

from models.pretrainedModel.pretrainmodels import get_resnet101,Net,get_convnext_base,get_resnet18

from utiles.image_csv import read_my_dir
from dataload.dataset import load_img
import time
import os
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
plt.switch_backend('agg')
import tqdm
from dataload.dataset import load_img
from models.mydesignModel.my_mobilenetV1 import MobileNet
device='cuda:0' if torch.cuda.is_available() else 'cpu'
# device='cpu'

def test_write_info(info,info_dir):
    with open(os.path.join(info_dir,'info.txt'),'a+') as f:
        f.write(info+'\n')
        f.write('\n')

def test_write_info_head(info,info_dir):
    with open(os.path.join(info_dir,'info.txt'),'a+') as f:
        f.write("-"*50+'\n')
        f.write("-" * 50+'\n')
        f.write("-----------------"+info+"------------------"+'\n')
        f.write("-" * 50+'\n')
        f.write("-" * 50+'\n')
        f.write('\n')


def cam(model_path,model,pred_dir):

    model_pth = torch.load(
        model_path, device)
    model.load_state_dict(model_pth)
    model = model.eval().to(device)
    # model = model.to(device)
    cam_extractor = GradCAMpp(model)
    _, df_test = read_my_dir()
    # date = time.strftime("%Y-%m-%d")
    # pred_dir = os.path.join('/mnt/llz/log/', date, 'pred')

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    df_pred = pd.DataFrame()
    test_write_info_head(' test ', pred_dir)
    for idx, row in df_test.iterrows():
        img_path = row['path']
        img_pil = load_img(img_path, df_pred)
        input_img = torch.Tensor(img_pil).unsqueeze(0).to(device)
        pred_logits = model(input_img)
        pred_softmax = torch.functional.F.softmax(pred_logits, dim=1)

        top_1 = torch.topk(pred_softmax, 1)
        pred_ids = top_1[1].cpu().detach().numpy().squeeze()
        pred_id=pred_ids.item()
        if idx == 242:
            # print('pred_ids size : ',pred_softmax.squeeze(0).argmax().item())
            # print('pred softmax size is ',pred_softmax.size())
            # activation_map = cam_extractor(pred_softmax.squeeze(0).argmax().item(), pred_softmax)
            activation_map = cam_extractor(pred_id, pred_softmax)
            activation_map = activation_map[0][0].detach().cpu().numpy()
            print(activation_map.shape)
            # plt.imshow(activation_map)
            # plt.savefig(os.path.join(pred_dir,'{}-HEAT_MAP-epoch-{}.png'.format(img_path[-1],epoch)))

            img_pil_trans = np.uint8(img_pil.transpose(1, 2, 0))
            # print(' img_pil  : ', img_pil.shape)
            # print(' img_pil_trans  : ', img_pil_trans.shape)
            # print('type img_pil  : ', type(Image.fromarray(img_pil_trans)))
            # print('activation_map[0].squeeze(0) is ', type(Image.fromarray(activation_map)))
            result = overlay_mask(Image.fromarray(img_pil_trans), Image.fromarray(activation_map), alpha=0.7)
            plt.imshow(result)
            plt.axis('off')
            plt.title('{}\nPred:{} Show:{}'.format(img_path.split('/')[-1], pred_ids, row['cls_no']))
            plt.savefig(os.path.join(pred_dir, '{}-CAM-pth-{}.png'.format(img_path.split('/')[-1],model_path.split('/')[-1].split('.')[0])))

            print('saved CAM img {}'.format(row['path']))
            print('saved in ',pred_dir)
            return 0


def calc(epoch,model,pred_dir):
    _, df_test = read_my_dir()
    # date = time.strftime("%Y-%m-%d")
    # pred_dir = os.path.join('/mnt/llz/log/',date,'pred')
    pred_df_path=os.path.join(pred_dir, 'df_pred.csv')
    classes = np.unique(df_test['cls_name'])
    print("classes name ",classes)
    df_pred = pd.DataFrame()


    for idx, row in df_test.iterrows():

        img_path = row['path']
        img_pil=load_img(img_path, df_pred)
        input_img = torch.Tensor(img_pil).unsqueeze(0).to(device)
        pred_logits = model(input_img)
        pred_softmax = torch.functional.F.softmax(pred_logits, dim=1)

        top_1 = torch.topk(pred_softmax, 1)
        pred_ids = top_1[1].cpu().detach().numpy().squeeze()
        pred_dict = {}
        pred_dict['pred'] = pred_ids
        pred_dict['pred_correct'] = row['cls_no'] == pred_ids
        for idx_cls, each in enumerate(classes):
            pred_dict['{}-pred_confidence_interval'.format(each)] = pred_softmax[0][idx_cls].cpu().detach().numpy()
            # print('{}-pred_confidence_interval'.format(each))
            # print(pred_softmax[0][idx_cls].cpu().detach().numpy())

        print(idx, ' -------> ', pred_ids, " ", pred_dict['npc-pred_confidence_interval'])
        test_write_info('{} \' --------> \' {} {}'.format(idx,pred_ids,pred_dict['npc-pred_confidence_interval']),pred_dir)
        df_pred = df_pred.append(pred_dict, ignore_index=True)
    df = pd.concat([df_test, df_pred], axis=1)

    acc=sum(df['pred']==df['cls_no'])/len(df)
    print('准确率为',acc)
    test_write_info_head('indices',pred_dir)
    test_write_info('pred acc is {}'.format(acc),pred_dir)
    df['cls_no'] = df['cls_no'].astype('int')
    df['pred'] = df['pred'].astype('int')
    df['npc-pred_confidence_interval'] = df['npc-pred_confidence_interval'].astype('float64')
    df.to_csv(pred_df_path, index=False)
    print(classification_report(df['cls_no'], df['pred'], target_names=['npc','non-npc']))
    test_write_info(classification_report(df['cls_no'], df['pred'], target_names=classes),pred_dir)
    specific_class = 'npc'
    y_test = (df['cls_name'] == specific_class)
    y_score = df['npc-pred_confidence_interval']
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, linewidth=5, label=specific_class)
    plt.plot([0, 1], [0, 1], ls="--", c='.3', linewidth=3, label='random')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.rcParams['font.size'] = 22
    plt.title('{} ROC  AUC:{:.3f}'.format(specific_class, auc(fpr, tpr)))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(pred_dir,'{}-ROC_IMG-epoch-{}.pdf'.format(specific_class,epoch)), dpi=120, bbox_inches='tight')
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    AP = average_precision_score(y_test, y_score, average='weighted')
    print('AP is ',AP)
    plt.figure(figsize=(12, 8))
    # 绘制 PR 曲线
    plt.plot(recall, precision, linewidth=5, label=specific_class)

    # 随机二分类模型
    # 阈值小，所有样本都被预测为正类，recall为1，precision为正样本百分比
    # 阈值大，所有样本都被预测为负类，recall为0，precision波动较大
    plt.plot([0, 0], [0, 1], ls="--", c='.3', linewidth=3, label='random')
    plt.plot([0, 1], [0.5, sum(y_test == 1) / len(df)], ls="--", c='.3', linewidth=3)

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.rcParams['font.size'] = 22
    plt.title('{} PR  AP:{:.3f}'.format(specific_class, AP))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(pred_dir,'{}-PR_IMG-{}.pdf'.format(specific_class,epoch)), dpi=120, bbox_inches='tight')




def sne_feature(model,pred_dir):
    from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
    import random
    import seaborn as sns
    from sklearn.manifold import TSNE
    plt.switch_backend('agg')
    model_trunc = create_feature_extractor(model, return_nodes={'classification_head1.3.avgpool': 'semantic_feature'})
    encoding_array = []
    img_path_list = []

    # date = time.strftime("%Y-%m-%d")
    # pred_dir = os.path.join('/mnt/llz/log/',date,'pred')
    # df_path =os.path.join(pred_dir,'df_pred.csv')


    df=pd.read_csv(df_path)
    class_list = np.unique(df['cls_name'])
    if not os.path.exists(os.path.join(pred_dir, 'test_sematic_characters.npy')):
        for img_path in df['crop_path']:
            img_path_list.append(img_path)

            input_img = torch.Tensor(load_img(img_path,df)).unsqueeze(0).to(device)
            feature = model_trunc(input_img)[
                'semantic_feature'].squeeze().detach().cpu().numpy()  # 执行前向预测，得到 avgpool 层输出的语义特征
            encoding_array.append(feature)
            print(img_path)
        encoding_array = np.array(encoding_array)
        np.save(os.path.join(pred_dir, 'test_sematic_characters.npy'), encoding_array)
    else:
        encoding_array=np.load(os.path.join(pred_dir, 'test_sematic_characters.npy'))
    print('encoding_array.shape is : ',encoding_array.shape)



    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x',
                   'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    n_class = len(class_list)  # 测试集标签类别数
    palette = sns.hls_palette(n_class)  # 配色方案

    # sns.palplot(palette)

    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)
    tsne = TSNE(n_components=2, n_iter=20000)
    X_tsne_2d = tsne.fit_transform(encoding_array)
    show_feature = 'cls_name'
    plt.figure(figsize=(14, 14))

    for idx, cls in enumerate(class_list):  # 遍历每个类别
        # 获取颜色和点型
        color = palette[idx]
        marker = marker_list[idx % len(marker_list)]

        # 找到所有标注类别为当前类别的图像索引号
        indices = np.where(df[show_feature] == cls)
        plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=cls, s=150)

    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(pred_dir,'t-SNE_2_features_show.pdf'), dpi=300)  # 保存图像
    tsne3 = TSNE(n_components=3, n_iter=10000)
    X_tsne_3d = tsne3.fit_transform(encoding_array)
    show_feature = '标注类别名称'
    # show_feature = '预测类别'
    df_3d = pd.DataFrame()
    df_3d['X'] = list(X_tsne_3d[:, 0].squeeze())
    df_3d['Y'] = list(X_tsne_3d[:, 1].squeeze())
    df_3d['Z'] = list(X_tsne_3d[:, 2].squeeze())
    df_3d['标注类别名称'] = df['cls_name']
    df_3d['预测类别'] = df['预测']
    df_3d['图像路径'] = df['crop_path']
    df_3d.to_csv(os.path.join(pred_dir,'t-SNE-3D.csv'), index=False)
    import plotly.express as px
    fig = px.scatter_3d(df_3d,
                        x='X',
                        y='Y',
                        z='Z',
                        color=show_feature,
                        labels=show_feature,
                        symbol=show_feature,
                        hover_name='图像路径',
                        opacity=0.6,
                        width=1000,
                        height=800)

    # 设置排版
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    fig.write_html(os.path.join(pred_dir,'语义特征t-SNE三维降维plotly可视化.html'))



def show_sne_feature(model_path):
    # model_path = '/mnt/llz/log/2022-02-22/save_models/fold_0_epoch_2_test_95.pt'
    model, model_name = get_resnet101()
    net = Net(model, model_name).to(device)
    net_pth = torch.load(
        model_path, device)
    net.load_state_dict(net_pth)
    model = net.eval().to(device)
    sne_feature(model)

if __name__ == '__main__':
    models_path='/mnt/llz/log/2023-03-27/distill/save_models'
    date = time.strftime("%Y-%m-%d")
    pred_dir = os.path.join('/mnt/llz/log/', date,"distill", 'pred')
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    # models_pth_list=os.listdir(models_path)
    model, model_name = get_resnet18()
    # model, model_name = get_resnet101()
    model = Net(model, model_name).to(device)
    # model = MobileNet().to(device)
    # model_name = 'my mobile net '

    # for path  in models_pth_list:
    #     model_path=os.path.join(models_path,path)
    #     print(model_path,'   ------> ',model_name)
    #     cam(model_path,model)
    # path='fold_0_epoch_0_94.pt'
    # path = 'fold_0_epoch_1_test_96.pt'
    # path='fold_0_epoch_4_test_97.pt'
    path='fold_0_student_epoch_58_99.pt'
    model_path = os.path.join(models_path, path)
    model_pth = torch.load(
        model_path, device)
    model.load_state_dict(model_pth)
    model =model .eval().to(device)
    calc(27, model,pred_dir)