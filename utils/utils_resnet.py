import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = 'cuda'


# Helper function to get image tensor
def get_img_tensor(image_rgb):
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image_rgb).unsqueeze(0).to(device)
    return img_tensor


# Helper function to get result (top prediction index and class)
def get_result(output):
    top_index = output.argmax().item()
    return top_index


def get_heatmap(top_index, output, cam_extractor, output_shape):
    """
    Generate a heatmap for a given top index using the CAM extractor and scale it to match the output dimension.

    Args:
        top_index (int): The index for which the activation map is generated.
        output (torch.Tensor): The output tensor from the model.
        cam_extractor (callable): The class activation mapping extractor function.

    Returns:
        np.ndarray: Scaled heatmap matching the output dimensions.
    """
    activation_map = cam_extractor(top_index, output)
    heatmap = activation_map[0][0].cpu().numpy()

    # Resize heatmap to match the output dimensions
    heatmap = cv2.resize(heatmap, (output_shape[1], output_shape[0]), interpolation=cv2.INTER_NEAREST)

    return heatmap


# Function to process image and extract CAM
def process_image_with_cam(image_rgb, heatmap):
    # Create a figure for displaying the image with heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(image_rgb)
    plt.imshow(heatmap, cmap='Reds', alpha=0.3, extent=[0, image_rgb.shape[1], image_rgb.shape[0], 0])
    plt.axis('off')
    plt.show()
