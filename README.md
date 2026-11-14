# Automatic Image Segmentation by Guiding SAM with Saliency Maps

This project proposes an automatic image segmentation by combining Saliency Maps with the Segment Anything Model (SAM). Utilizing a pretrained classification model, one can generate predictions and saliency maps using existing XAI methods, which are then employed to guide SAM segmentation.

<!-- Tasks:
- Literature Review: Understand and summarize existing work in XAI and applications of SAM.
- Incorporate saliency maps and predictions to guide SAM.
- Evaluate the approach based on suitable metrics.
- Integrate the approach into Label Studio.
- Write a scientific report on your experimental results.  -->

## Evaluator Pipeline

The evaluator pipeline is a Python script that evaluates the performance of the proposed method on a dataset. The pipeline consists of the following steps:

- Load the dataset
- Load the pretrained classification model
- Generate predictions and saliency maps
- Apply SAM segmentation
- Evaluate the segmentation performance

## Handlers

The handlers are Python scripts that define the logic for certain parts of the pipeline. The handlers are:

- `dataset_handler.py`: Contains functions to load the dataset.
- `model_handler.py`: Contains functions to load the pretrained classification model.
- `heatmap_handler.py`: Contains functions to generate saliency maps.
- `prompt_handler.py`: Contains functions to guide SAM segmentation.
- `segment_eval_handler.py`: Contains functions to evaluate the segmentation performance.

These abstract handlers can be implemented in different ways depending on the specific requirements of the project.
