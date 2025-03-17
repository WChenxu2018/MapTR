import torch
import mmdet
import mmcv
import mmdet3d
print("mmcv", mmcv.__version__)
print("torch", torch.__version__)
print("cuda.version", torch.version.cuda)
print("mmdet", mmdet.__version__)
print("mmdet3d ", mmdet3d.__version__)

 
print(torch.__version__)  # 查看torch当前版本号
print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True，则表示可用
print(torch.backends.cudnn.version()) # 查看cudnn版本是否正确
print(torch.backends.cudnn.is_available())
