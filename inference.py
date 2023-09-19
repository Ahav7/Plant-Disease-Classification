import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import argparse
import pathlib

from model import build_model
from class_names import class_names as CLASS_NAMES

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights',
    default='../outputs/efficientnet_80_20/best_model.pth',
    help='path to the model weights',
)
parser.add_argument(
    '-i', '--image',
    default='../input/inference_data/apple_scab.jpg',
    help='path to the model weights',
)
args = vars(parser.parse_args())

# Constants and other configurations.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224


# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform


def denormalize(
        x,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)


def annotate_image(image, output_class):
    image = denormalize(image).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    class_name = CLASS_NAMES[int(output_class)]
    # The class name consists of the plant name and the 
    # disease.
    plant = class_name.split('___')[0]
    disease = class_name.split('___')[-1]
    cv2.putText(
        image,
        f"{plant}",
        (5, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA
    )
    cv2.putText(
        image,
        f"{disease}",
        (5, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        lineType=cv2.LINE_AA
    )
    return image


def infer(model, testloader, DEVICE):
    """
    Function to run inference.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.
    """
    model.eval()
    counter = 0
    with torch.no_grad():
        counter += 1
        image = testloader
        image = image.to(DEVICE)

        # Forward pass.
        outputs = model(image)
    # Softmax probabilities.
    predictions = F.softmax(outputs, dim=1).cpu().numpy()
    # Predicted class number.
    output_class = np.argmax(predictions)
    # Show and save the results.
    # result = annotate_image(image, output_class)
    class_name = CLASS_NAMES[int(output_class)]
    # The class name consists of the plant name and the
    # disease.
    plant = class_name.split('___')[0]
    disease = class_name.split('___')[-1]
    return plant, disease


# if __name__ == '__main__':
#     weights_path = pathlib.Path(args['weights'])
#     model_name = str(weights_path).split(os.path.sep)[-2]
#     print(model_name)
#     infer_result_path = os.path.join('..', 'outputs', 'inference_results', model_name)
#     os.makedirs(infer_result_path, exist_ok=True)
#     img_path = args['image']
#     checkpoint = torch.load(weights_path)
#     # Load the model.
#     model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])

    # all_image_paths = glob.glob(os.path.join('..', 'input', 'inference_data', '*'))
    #
    # transform = get_test_transform(IMAGE_RESIZE)
    #
    # for i, image_path in enumerate(all_image_paths):
    #     print(f"Inference on image: {i + 1}")
    #     image = cv2.imread(image_path)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = transform(image)
    #     image = torch.unsqueeze(image, 0)
    #     result = inference(
    #         model,
    #         image,
    #         DEVICE
    #     )
    #     # Save the image to disk.
    #     image_name = image_path.split(os.path.sep)[-1]
    #     cv2.imshow('Image', result)
    #     cv2.waitKey(1)
    #     cv2.imwrite(os.path.join(infer_result_path, image_name), result * 255.)

    # For inferencing a single image
    # print(f"Inference on image located at: {img_path}")
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = transform(image)
    # image = torch.unsqueeze(image, 0)
    # result = infer(model, image, DEVICE)
    # # Save the image to disk.
    # image_name = img_path.split(os.path.sep)[-1]
    # cv2.imshow('Image', result)
    # cv2.waitKey(1)
    # cv2.imwrite(os.path.join(infer_result_path, image_name), result * 255.)
