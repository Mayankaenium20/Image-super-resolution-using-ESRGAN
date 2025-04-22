import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

#Model and device setup
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')  # Use CPU for inference

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob('LR/*'):  # Update the path to your test images folder
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)

    #read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)

    #perform inference
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().clamp_(0, 1).numpy()

    #img conversion for saving
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
