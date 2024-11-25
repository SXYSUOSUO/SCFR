from maskrcnn_benchmark.data import datasets

from .coco import coco_evaluation
from .voc import voc_evaluation
import pdb

def evaluate(dataset, predictions, output_folder, **kwargs):
    """evaluate dataset using different methods based on dataset type.
    Args:
        dataset: Dataset object
        predictions(list[BoxList]): each item in the list represents the
            prediction results for one image.
        output_folder: output folder, to save evaluation files or results.
        **kwargs: other args.
    Returns:
        evaluation result
    """
    
    #if isinstance(dataset, datasets.MultiDataset):
    #    dataset = dataset.datasets[0]
    args = dict(dataset=dataset, predictions=predictions, output_folder=output_folder, **kwargs)
    #pdb.set_trace()
    if isinstance(dataset, datasets.COCODataset):
        return coco_evaluation(**args)
    elif isinstance(dataset, datasets.PascalVOCDataset):
        return voc_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}.".format(dataset_name))
