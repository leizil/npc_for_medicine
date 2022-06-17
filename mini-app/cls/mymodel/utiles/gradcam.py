import torch
import numpy as np
import PIL.Image as Image

class CamExtractor():
    def __init__(self,model):
        self.model=model
        self.gradients=[]

    def save_gradient(self,grad):
        self.gradients.append(grad)

    def forward_pass_on_convolutions(self,x):
        conv_output=[]
        x=self.model.conv1(x)
        x=self.model.bn1(x)
        x=self.model.relu(x)
        x=self.model.maxpool(x)

        x=self.model.layer1(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x=self.model.layer2(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x=self.model.layer3(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        x=self.model.layer4(x)
        x.register_hook(self.save_gradient)
        conv_output.append(x)

        return conv_output,x

    def forward_pass(self,x):
        conv_output,x=self.forward_pass_on_convolutions(x)

        x=self.model.avgpool(x)
        x=torch.flatten(x,1)
        x=self.model.fc(x)
        return conv_output,x


class GradCam():
    def __init__(self,model):
        self.model=model
        self.model.eval()
        self.extractor=CamExtractor(self,model)

    def generate_cam(self,input_image,target_layer,target_class=None):
        conv_output,model_output=self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())

        one_hot_output=torch.FloatTensor(1,model_output.size()[-1]).zero_()
        one_hot_output[0][target_class]=1

        self.model.zero_grad()

        model_output.backward(gradient=one_hot_output,retain_graph=True)

        guided_gradients=self.extractor.gradients[-1-target_layer].data.numpy()[0]

        target=conv_output[target_layer].data.numpy()[0]
        weights=np.mean(guided_gradients,axis=(1,2))

        cam=np.ones(target.shape[1:],dtype=np.float32)

        for i,w in enumerate(weights):
            cam+=w*target[i,:,:]
        cam=np.maximum(cam,0)
        cam=(cam-np.min(cam))/(np.max(cam)-np.min(cam))
        cam=np.uint8(cam*255)
        cam_resize = Image.fromarray(cam).resize((input_image.shape[2],
                                                  input_image.shape[3]),
                                                 Image.ANTIALIAS)
        cam=np.uint8(cam_resize) /255

        return cam


