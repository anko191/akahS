import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定

import torchvision as T

def get_transforms():
    Tra = T.transforms.Compose([
                     T.transforms.Resize((224,244)),
                     T.transforms.ToTensor(),
                     T.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                     ])
    return Tra
