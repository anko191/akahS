import base64

from django.shortcuts import render, redirect
from django.views import generic
# import numpy as np
from PIL import Image
import numpy as np
import torch
from Net.get_transforms import get_transforms

txt = {0:' ゆず ', 1:' むぎ ', 2:' とと ', 3:' もっぷ '}

# from cnn.simple_convnet import SimpleConvNet
from Net.EfficientNet import EfficientNet

model = EfficientNet(4, model_name = "tf_efficientnet_b0")
model.load_state_dict(torch.load('SHAKA.pth', map_location = "cpu"))
model.eval()

txt = {0:'ゆず', 1:'むぎ', 2:'とと', 3:'もっぷ'}


def shaka(request):
    return render(request, 'main/home.html', {})

def upload(request):
    file = request.FILES.getlist("one_file")
    if len(file) == 0:
        return redirect('home')
    file = file[0]
    if request.method == 'POST' and file:
        img = Image.open(file)
        if img.mode != 'RGB':
            return render(request, 'main/result.html', {'result':[['None','カラー画像でお願いします。']]})
        img = get_transforms()(img)
        pred = model(img[np.newaxis,...])
        _, ans = torch.max(pred, 1)
        # for file, label in zip(files, labels):
        file.seek(0)
        src = base64.b64encode(file.read()).decode()
        #print(src)
        result = []
        # django2.0からは src = base64.b64encode(file.read()).decode()
        result.append((src, txt[ans.to("cpu").detach().numpy().copy()[0]], pred.to("cpu").detach().numpy().copy().squeeze(0)))
        context = {
            'result': result,
        }
        return render(request, 'main/result.html', context)
    else:
        return redirect('home')

# def upload(request):
#     files = request.FILES.getlist("files[]")
#     if request.method == 'POST' and files:
#         array_list = []
#         for file in files:
#             img = Image.open(file)
#             if img.mode != 'RGB':
#                 return render(request, 'main/result.html', {'result':[['None','カラー画像でお願いします。']]})
#             array = np.asarray(img)
#             array_list.append(array)
#
#         x = np.array(array_list)
#         #labels = network.predict(x).argmax(axis=1)
#         imgs = get_transforms(data_config)(img)
#         labels = model(imgs)
#         result = []
#         for file, label in zip(files, labels):
#             file.seek(0)
#             src = base64.b64encode(file.read()).decode()
#             #print(src)
#             # django2.0からは src = base64.b64encode(file.read()).decode()
#             result.append((src, label))
#         context = {
#             'result': result,
#         }
#         return render(request, 'main/result.html', context)
#     else:
#         return redirect('home')
