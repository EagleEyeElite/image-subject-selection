class Prompt():
    def __init__(self) -> None:  # TODO: ad image id
        self.original_image_id = None
        self.original_image = None
        self.original_image_tensor = None
        self.original_image_class = None
        self.mask = None
        self.sam_predictor = None
        self.predicted_class_id = None
        self.predicted_class_label = None
        self.heatmap = None
        self.points = None
        self.point_labels = None
        self.bounding_boxes = None
        self.text_prompt = None
        self.segmentation_count = None
        self.segmentation_masks = None
        self.segmentation_scores = None
        self.segmentation_logits = None
        self.segmentation_evaluation = [-1, -1, -1]

    def __str__(self) -> str:
        dict_str = {
            "original_image": self.original_image.shape if self.original_image is not None else None,
            "original_image_tensor": self.original_image_tensor.shape if self.original_image_tensor is not None else None,
            "original_image_class": self.original_image_class,
            "mask": self.mask.shape if self.mask is not None else None,
            "sam_predictor": self.sam_predictor is not None,
            "predicted_class_id": self.predicted_class_id,
            "predicted_class_label": self.predicted_class_label,
            "heatmap": self.heatmap.shape if self.heatmap is not None else None,
            "points": self.points,
            "point_labels": self.point_labels,
            "bounding_boxes": self.bounding_boxes,
            "text_prompt": self.text_prompt,
            "segmentation_count": self.segmentation_count,
            "segmentation_masks": self.segmentation_masks is not None,
            "segmentation_scores": self.segmentation_scores,
            "segmentation_logits": self.segmentation_logits is not None,
            "segmentation_evaluation": self.segmentation_evaluation,
        }
        return ("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in dict_str.items()) + "}")

    def cleanup(self):
        # self.original_image = None
        # self.original_image_tensor = None
        # self.original_image_class = None
        # self.mask = None
        # self.sam_predictor = None
        # self.predicted_class_id = None
        # self.predicted_class_label = None
        # self.heatmap = None
        # self.points = None
        # self.point_labels = None
        # self.bounding_boxes = None
        # self.text_prompt = None
        # self.segmentation_count = None
        # self.segmentation_masks = None
        # self.segmentation_scores = None
        # self.segmentation_logits = None
        # self.segmentation_evaluation = [-1, -1, -1]
        del self.original_image
        del self.original_image_tensor
        del self.original_image_class
        del self.mask
        del self.sam_predictor
        del self.predicted_class_id
        del self.predicted_class_label
        del self.heatmap
        del self.points
        del self.point_labels
        del self.bounding_boxes
        del self.text_prompt
        del self.segmentation_count
        for mask in self.segmentation_masks:
            del mask
        for score in self.segmentation_scores:
            del score
        for logit in self.segmentation_logits:
            del logit
        del self.segmentation_evaluation
        return self
