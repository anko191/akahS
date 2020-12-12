import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import timm
import torch.nn as nn



class EfficientNet(nn.Module):
    def __init__(self, n_out, model_name):
        super(EfficientNet, self).__init__()
        self.effnet = timm.create_model(model_name, pretrained = True)
        self.effnet.classifier = nn.Linear(1280, n_out)
    def forward(self, x):
        return self.effnet(x)
