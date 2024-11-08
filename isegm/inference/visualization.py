import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk, circle_perimeter
from pathlib import Path
import cv2

def draw_qualitative_example(image, clicks_list, pred_mask, gt_mask,
    saturation_decay = 0.3, mask_color=[0.1, 0.1, 0.9],
    mask_opacity=0.3, pos_point_color=[0.,1.,0.], neg_point_color=[1.,0.,0.],
    point_radius=4,ring_thickness=2, dataset_name=None, sample_id=None,
    visualization_path=None
    ):

    # Numpy conversions
    mask_color = np.array(mask_color)
    pos_point_color = np.array(pos_point_color)
    neg_point_color = np.array(neg_point_color)

    # 1. Decrease saturation for better visibility
    image_gray = image.mean(axis=2, keepdims=True)
    image = saturation_decay*image_gray + (1.-saturation_decay)*image

    # 2. Plot predicted mask to the image
    image_pred = plot_mask(image, pred_mask, mask_color, mask_opacity)

    # 3. Draw the points to the image
    for cl in clicks_list:
        color = pos_point_color if cl.is_positive else neg_point_color
        # First draw a slightly larger whiter circle to serve as a boundary
        # ring for better visualization
        image_pred = draw_point_autosize(
            img=image_pred, x=cl.coords[1], y=cl.coords[0],
            color=np.array([1.,1.,1.]), factor=0.02
        )
        image_pred = draw_point_autosize(
            img=image_pred, x=cl.coords[1], y=cl.coords[0],
            color=color, factor=0.017
        )

    # 4. Draw the ground truth mask
    image_gt = plot_mask(image, gt_mask, mask_color, mask_opacity)


    ############################
    #fig, ax = plt.subplots(2, 1)
    #ax[0].imshow(image_pred)
    #ax[1].imshow(image_gt)
    #plt.show()
    folder = Path(visualization_path) / dataset_name
    folder.mkdir(parents=True, exist_ok=True)

    gt_name = str(folder / "{:05d}_gt_plot.png".format(sample_id))
    pred_name = str(folder / "{:05d}_pred_plot.png".format(sample_id))

    cv2.imwrite(gt_name, image_gt[:, :, ::-1] * 255.0)
    cv2.imwrite(pred_name, image_pred[:, :, ::-1] * 255.0)



def draw_point_autosize(img, x, y, color, factor=0.02):
    height = img.shape[0]
    x, y = int(np.floor(x)), int(np.floor(y))
    radius = int(np.floor(factor * height))

    drr, dcc = disk((y, x), radius)
    crr, ccc = circle_perimeter(y, x, radius)

    img = np.copy(img)

    # Prune the drawings
    drr_new, dcc_new = [], []
    for dr, dc in zip(drr, dcc):
        if dr < img.shape[0] and dc < img.shape[1]:
            drr_new.append(dr)
            dcc_new.append(dc)
    drr, dcc = np.array(drr_new), np.array(dcc_new)

    crr_new, ccc_new = [], []
    for cr, cc in zip(crr, ccc):
        if cr < img.shape[0] and cc < img.shape[1]:
            crr_new.append(cr)
            ccc_new.append(cc)
    crr, ccc = np.array(crr_new), np.array(ccc_new)

    img[drr, dcc] = color
    img[crr, ccc] = np.array([1., 1., 1.])

    return img

def plot_mask(img, mask, color, mask_opacity=0.5):
    img_plot = np.where(
        mask[..., np.newaxis] == 1,
        img * mask_opacity + (1. - mask_opacity) * color,
        img
    )
    return img_plot