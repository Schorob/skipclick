import torch
import torch.nn as nn
from isegm.inference import utils
import numpy as np



def set_image_demo(model, image_np, internal_size=(448, 448), squeeze=True):
    model.internal_size = internal_size
    model.original_size = image_np.shape[:2]
    device_idx = model.fuse.weight.get_device()
    device = torch.device("cuda:" + str(device_idx) if device_idx >= 0 else "cpu")
    image = np.transpose(image_np, (2, 0, 1)) / 255.0
    image = torch.tensor(image).float().to(device).unsqueeze(0)
    image = nn.functional.interpolate(image, size=model.internal_size, mode='bilinear', align_corners=True)
    model.set_image(image)


def apply_click_demo(model, points, labels, prev_mask):
    # print("Input from GUI on click: ",points, labels, prev_mask.shape)
    points = [(x, y, i) for i, (x, y) in enumerate(points)]
    pos_points = [p for p, l in zip(points, labels) if l == 1]
    neg_points = [p for p, l in zip(points, labels) if l == 0]
    max_point_num = max(len(pos_points), len(neg_points))
    points_ritm = np.full((1, max_point_num * 2, 3), -1., dtype=np.float32)

    for i, p in enumerate(pos_points):
        points_ritm[0, i, 1] = p[0] / model.original_size[1] * model.internal_size[1]
        points_ritm[0, i, 0] = p[1] / model.original_size[0] * model.internal_size[0]
        points_ritm[0, i, 2] = p[2]

    for i, p in enumerate(neg_points):
        points_ritm[0, max_point_num + i, 1] = p[0] / model.original_size[1] * model.internal_size[1]
        points_ritm[0, max_point_num + i, 0] = p[1] / model.original_size[0] * model.internal_size[0]
        points_ritm[0, max_point_num + i, 2] = p[2]

    device_idx = model.fuse.weight.get_device()
    device = torch.device("cuda:" + str(device_idx) if device_idx >= 0 else "cpu")
    points_ritm = torch.tensor(points_ritm).float().to(device)

    prev_mask = torch.tensor(prev_mask).float().to(device).unsqueeze(0).unsqueeze(0)
    prev_mask_squeezed = nn.functional.interpolate(prev_mask, size=model.internal_size, mode='bilinear',
                                                   align_corners=True)

    predicted_mask = model.apply_click(prev_mask_squeezed, points_ritm)
    predicted_mask = torch.sigmoid(predicted_mask)
    predicted_mask = nn.functional.interpolate(predicted_mask, size=model.original_size, mode='bilinear',
                                               align_corners=True)
    predicted_mask = predicted_mask[0, 0].detach().cpu().numpy()

    return predicted_mask