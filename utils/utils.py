from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from dataset_handler.dataset_handler import DatasetHandler
from heatmap_handler.heatmap_handler import HeatmapHandler
from model_handler.model_handler import ModelHandler
from tqdm import tqdm
import torchvision
# TODO: Typing and comments


class SquareCropAndResize(object):
    def __init__(self, size=None, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # Calculate the center crop size
        w, h = img.size
        new_size = min(w, h)

        # Center crop the image to a square
        left = (w - new_size) // 2
        top = (h - new_size) // 2
        right = (w + new_size) // 2
        bottom = (h + new_size) // 2
        img = img.crop((left, top, right, bottom))

        # Resize the image to the specified size
        if self.size is not None:
            img = img.resize((self.size, self.size), self.interpolation)

        return img


def sam_predict(entry, multiple_results=False):
    masks, scores, logits = entry.sam_predictor.predict(
        point_coords=entry.points,
        point_labels=entry.point_labels,
        multimask_output=multiple_results
    )

    entry.segmentation_masks = []
    entry.segmentation_scores = []
    entry.segmentation_logits = []
    entry.segmentation_count = len(scores)

    # sort by score
    sorted_idx = np.argsort(scores)[::-1]
    for indx in sorted_idx:
        entry.segmentation_masks.append(masks[indx])
        entry.segmentation_scores.append(scores[indx])
        entry.segmentation_logits.append(logits[indx])
    return entry


class visualizer:
    def __init__(self, device, model: ModelHandler, dataset: DatasetHandler, xai_methods: list[HeatmapHandler], sample_size: int) -> None:
        self.device = device
        self.model = model
        self.dataset = dataset
        self.xai_methods = xai_methods
        self.sample_size = sample_size
        self.samples = self.dataset.get_samples(sample_size)
        self.input_size = self.model.input_size
        # self.create_plot()

    def create_plot(self):
        self.fig, self.axs = plt.subplots(
            nrows=self.sample_size,
            ncols=len(self.xai_methods) + 1,  # +1 for the original image
            figsize=((len(self.xai_methods) + 1) * 2, 2 * self.sample_size),
        )

        # plt.tight_layout()

        for ax in self.axs.flatten():
            ax.axis("off")

        x = np.linspace(0, self.input_size, 8)
        y = np.linspace(0, self.input_size, 8)
        X, Y = np.meshgrid(x, y)

        for i, row in enumerate(tqdm(self.samples.itertuples(), total=self.sample_size)):
            img = self.dataset.get_PIL_image(row, self.input_size)
            img_tensor = self.dataset.get_image_tensor(row, self.input_size)
            mask = self.dataset.get_mask(row, self.input_size)

            if i == 0:
                self.axs[i, 0].title.set_text("Original image")

            self.axs[i, 0].imshow(img)

            # add vertical title left
            true_label = self.dataset.get_true_label(row)
            self.axs[i, 0].text(
                -0.1,
                0.5,
                f"True: {true_label}",
                size=12,
                ha="center",
                va="center",
                rotation=90,
                transform=self.axs[i, 0].transAxes,
            )

            for j, method_class in enumerate(self.xai_methods):
                m = method_class(self.model)
                name = m.name
                heatmap, pred_label = m(img_tensor)
                if i == 0:
                    self.axs[i, j + 1].title.set_text(name)
                heatmap = np.flipud(heatmap[0].cpu().numpy())
                self.axs[i, j + 1].pcolormesh(X, Y, heatmap,
                                              cmap='viridis', shading='auto')
                self.axs[i, j + 1].set_aspect('equal')
                self.axs[i, j + 1].contour(mask[::-1],
                                           levels=[0.5], colors='w', alpha=0.6)
                self.axs[i, j + 1].title.set_text(pred_label)

        plt.tight_layout()
        return self.fig


def evaluate_xai_methods(true_mask, heatmap, ds_idx, pred_label, true_label, xai_method, metrics):
    m = {
        "dataset_idx": ds_idx,
        "xai_method": xai_method,
        "true_label": true_label,
        "pred_label": pred_label,
    }

    for metric_name, metric in metrics.items():
        m[metric_name] = metric(heatmap, true_mask)

    return m

def min_max_normalize(mask):
    """
    Min-max normalization for each channel
    Args:
        mask (torch.Tensor): Mask CxWxH.

    """
    mins = mask.amin(dim=(-2, -1))
    maxs = mask.amax(dim=(-2, -1))
    # expand dims to [N activations, 1, 1]
    maxs, mins = maxs[:, None, None], mins[:, None, None]

    return (mask - mins) / (maxs - mins + 1e-8)


def img_to_cv2(image):
    """
    Convert PIL image to OpenCV format.
    
    Args:
    - image: PIL Image.
    
    Returns:
    - image_cv2: OpenCV image (NumPy array).
    """
    return np.array(image)

imagenet_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Resize segmentation masks the same way as imagenet_transform resizes images
# (nearest neighbour so class ids stay intact).
mask_transform = torchvision.transforms.Resize(
    224, interpolation=torchvision.transforms.InterpolationMode.NEAREST
)

square_transform = torchvision.transforms.Compose(
    [
        SquareCropAndResize(size=None),
    ]
)

inverse_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        torchvision.transforms.Normalize(
            mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
        ),
    ]
)

target_transform = SquareCropAndResize(224, interpolation=Image.NEAREST)