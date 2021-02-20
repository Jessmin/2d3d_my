import torch
from model.sttn.model.sttn import InpaintGenerator
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# cap = torch.cuda.memory_allocated(device)
print(torch.cuda.get_device_properties(device))
print(torch.cuda.get_device_properties(device).total_memory)
#
# net = InpaintGenerator()
# model = net.to(device)
# ckpt = '../model/sttn/checkpoints/sttn.pth'
# data = torch.load(ckpt, map_location=device)
# model.load_state_dict(data['netG'])
# model.eval()
# # 875Ml
import numpy as np

feats = np.random.random((70, 432 * 2, 240))
feats = torch.from_numpy(feats)
print(type(feats))
feats.to(device)
# masks.to(device)
print('load Done')

while True:
    x = 1
