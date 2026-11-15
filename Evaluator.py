
# Imports
from model_handler.resnet import resnet
from dataset_handler.pascal import PascalDataset
import heatmap_handler
from model_handler import device
from utils.utils import SquareCropAndResize, imagenet_transform, inverse_transform, mask_transform, sam_predict
import prompt_handler
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os
import segment_eval_handler
from segment_eval_handler.iou import IOU
from utils.eval_visualizer import EvalVisualizer
from tqdm import tqdm
from prompt_handler.max_value import MaxValue
import numpy as np
import pandas as pd
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Handler Loader
model = resnet(4)
dataset = PascalDataset(device, imagenet_transform,
                        inverse_transform, target_transform=mask_transform)
# currently returns only the methods combined with negative grid
xai_methods = heatmap_handler.get_lrp_methods()
point_generator = prompt_handler.get_best_methods()
# point_generator = [point_generators[0]]
segment_eval = [IOU]


# Instantiate Lists
for indx, method in enumerate(xai_methods):
    xai_methods[indx] = method(model)


for indx, method in enumerate(point_generator):
    point_generator[indx] = method()

for indx, method in enumerate(segment_eval):
    segment_eval[indx] = method()


# SAM
sam_checkpoint = os.environ.get('SAM_CHECKPOINT', 'checkpoints/sam_vit_b_01ec64.pth')
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam = sam.to(device)
print(f"Device: {device}")


# Initialize data


data = {}
for indx_i, i in enumerate(xai_methods):
    xai_name = xai_methods[indx_i].__class__.__name__
    xai_name = xai_methods[indx_i].instance_name if hasattr(
        xai_methods[indx_i], 'instance_name') else xai_name
    data[xai_name] = {}
    for indx_j, j in enumerate(point_generator):
        point_name = point_generator[indx_j].__class__.__name__
        data[xai_name][point_name] = {}
        for indx_k, k in enumerate(segment_eval):
            segment_name = segment_eval[indx_k].__class__.__name__
            data[xai_name][point_name][segment_name] = []


COUNT = 2
NORMALIZE_HEATMAP = True
rows = dataset.get_all_rows(COUNT)
log = True
df = []


for i in tqdm(rows, total=COUNT):
    # Get Prompt
    prompt = dataset.get_prompt(i)
    # Initialize SAM Embedding
    predictor = SamPredictor(sam)
    predictor.set_image(prompt.original_image)
    prompt.sam_predictor = predictor
    for method in xai_methods:
        xai_name = method.__class__.__name__
        xai_name = method.instance_name if hasattr(
            method, 'instance_name') else xai_name
        if log:
            print(f"Method: {xai_name}")
            print("-------------------------")

        method(prompt)
        # normalize to sum to 1 in range 0-1
        if NORMALIZE_HEATMAP:
            heatmap = prompt.heatmap
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / heatmap.max()
            prompt.heatmap = heatmap

        for point in point_generator:
            point_name = point.__class__.__name__
            if log:
                print(f"Method: {xai_name}")
                print(f"Point Method: {point_name}")
                print("-------------------------")

            try:
                point(prompt)
            except Exception as e:
                # set single point to 0, 0, true
                prompt.points = np.array([(0, 0)])
                prompt.point_labels = np.array([1])

            sam_predict(prompt, True)
            pd_xai_name = f'{xai_name}___{point_name}'
            row_data = {
                'dataset_idx': prompt.original_image_id,
                'xai_method': xai_name,
                # Assuming true_label is based on original image class
                'true_label': prompt.original_image_class,
                # Assuming predicted label is from prompt
                'pred_label': prompt.predicted_class_label,
            }
            evals = {}
            for eval_func in segment_eval:
                eval_name = eval_func.__class__.__name__
                if log:
                    print(f"Segmentation Evaluation: {eval_name}")
                eval_func(prompt)
                data[xai_name][point_name][eval_name].append(
                    prompt.segmentation_evaluation)
                evals[f'{eval_name}_0'] = prompt.segmentation_evaluation[0]
                evals[f'{eval_name}_max'] = max(prompt.segmentation_evaluation)
                if log:
                    print("-------------------------")
            row_data.update(evals)
            df.append(row_data)

    # cleanup prompt and predictor
    prompt.cleanup()
    del predictor
    del prompt
df = pd.DataFrame(df)


NAME = 'test1'


with open(f'{NAME}.json', 'w') as f:
    import json
    json.dump(data, f)
with open(f'{NAME}.json', 'r') as f:
    import json
    data = json.load(f)
EvalVisualizer(data)

df.to_csv(f'{NAME}.csv', index=False)
