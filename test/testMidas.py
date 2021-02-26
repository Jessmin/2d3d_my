import torch
import cv2
import numpy as np
import torch.autograd as autograd
from torchvision import transforms


def to_Img(data):
    data = 255 * (data - data.min()) / (data.max() - data.min())
    return data.astype(np.uint8)


import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
midas.to(device)
midas.eval()
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

calcDepthTen = torch.from_numpy(img)

calcDepthTen = autograd.Variable(calcDepthTen.cuda(device=device, non_blocking=True), requires_grad=False)

ch = calcDepthTen.shape[-1]
frmHeight = calcDepthTen.shape[0]
frmWidth = calcDepthTen.shape[1]
calcDepthTen = calcDepthTen.view(-1, ch).transpose(1, 0).contiguous().view(ch, frmHeight, frmWidth)
batchInput = []
batchInput.append(calcDepthTen)
batchInput = torch.stack(batchInput)
batchInput = batchInput.type(torch.float32)/255.
batchInput = transform(batchInput)

with torch.no_grad():
    prediction = midas(batchInput)

    # prediction = torch.nn.functional.interpolate(
    #     prediction.unsqueeze(1),
    #     size=img.shape[:2],
    #     mode="bicubic",
    #     align_corners=False,
    # ).squeeze()
    prediction = prediction.squeeze()
print(prediction.dtype)
output = prediction.cpu().numpy()
cv2.imshow('img', to_Img(output))
cv2.waitKey(0)
