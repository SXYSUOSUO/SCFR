# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time
import os

import torch
from tqdm import tqdm

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from maskrcnn_benchmark.data import datasets
import pdb


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = {}
    results_dict1 = {}
    results_dict2 = {}
    cpu_device = torch.device("cpu")
    for i, batch in enumerate(tqdm(data_loader)):
        #pdb.set_trace()
        if len(batch)==6:
            images1,targets1,image_ids,images2,targets2,_= batch
            images1 = images1.to(device)
            images2 = images2.to(device)
            with torch.no_grad():
                #pdb.set_trace()
                output1,output2 = model.forward2(images1,images2)
                output1 = [o.to(cpu_device) for o in output1]
                output2 = [o.to(cpu_device) for o in output2]
            results_dict1.update({img_id: result for img_id, result in zip(image_ids, output1)})
            results_dict2.update({img_id: result for img_id, result in zip(image_ids, output2)})
        elif len(batch)==3:
            images, targets, image_ids = batch
            images = images.to(device)
            with torch.no_grad():
                #pdb.set_trace()
                output = model(images)
                output = [o.to(cpu_device) for o in output]
            results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})
    if results_dict =={}:
        return [results_dict1,results_dict2]
    else:
        return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = (
        torch.distributed.get_world_size()
        if torch.distributed.is_initialized()
        else 1
    )
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    
    start_time = time.time()
    #pdb.set_trace()
    predictions = compute_on_dataset(model, data_loader, device)
    
    #pdb.set_trace()
    
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )

    if len(predictions)>1 and type(dataset) == datasets.multi_dataset.MultiDataset:
        for i in range(0,len(predictions)):
            predictions[i] = _accumulate_predictions_from_multiple_gpus(predictions[i])
    else:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    

    #pdb.set_trace()
    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
    if isinstance(dataset, datasets.MultiDataset):
        for i in range(0,len(dataset.datasets)):
            logger.info("Start evaluation on {} dataset({} images).".format(dataset_name[i], len(dataset)))
            if output_folder:
                torch.save(predictions[i], os.path.join(output_folder[i], "predictions.pth"))
            evaluate(dataset=dataset.datasets[i],predictions=predictions[i],output_folder=output_folder[i],**extra_args)

    else:
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
        if output_folder:
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        evaluate(dataset=dataset,predictions=predictions, output_folder=output_folder,**extra_args)
