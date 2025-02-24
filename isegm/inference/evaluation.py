from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker
from . import visualization

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        for object_id in sample.objects_ids:
            _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                sample_id=index, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None, dataset_name=None,
                    visualize_results=False, visualization_path=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break
        #print("clicker.clicks_list: ")
        #for cl in clicker.clicks_list:
        #    print("clickpos: ", cl.coords, "| clickval: ", cl.is_positive)
        if visualize_results:
            visualization.draw_qualitative_example(
                image=image, clicks_list=clicker.clicks_list,
                pred_mask=pred_mask, gt_mask=gt_mask, dataset_name=dataset_name,
                sample_id=sample_id, visualization_path=visualization_path
            )

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs
