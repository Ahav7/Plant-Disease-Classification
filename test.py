import torch
import numpy as np
import cv2
import os
import torchvision.transforms as transforms
import argparse
import pathlib

from tqdm.auto import tqdm
from model import build_model
from torch.utils.data import DataLoader
from torchvision import datasets
from class_names import class_names

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-w', '--weights',
    default='outputs/efficientnet/best_model.pth',
    help='path to the model weights',
)
args = vars(parser.parse_args())

# Constants and other configurations.
TEST_DIR = os.path.join('', 'input', 'test')
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
NUM_WORKERS = 4
CLASS_NAMES = class_names


# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return test_transform


def get_datasets(image_size):
    """
    Function to prepare the Datasets.
    Returns the test dataset.
    """
    dataset_test = datasets.ImageFolder(TEST_DIR, transform=(get_test_transform(image_size)))
    return dataset_test


def get_data_loader(dataset_test):
    """
    Prepares the training and validation data loaders.
    :param dataset_test: The test dataset.

    Returns the training and validation data loaders.
    """
    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    return test_loader


def denormalize(
        x,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)


def save_test_results(
        tensor,
        target,
        output_class,
        counter,
        test_result_save_dir
):
    """
    This function will save a few test images along with the 
    ground truth label and predicted label annotated on the image.

    :param tensor: The image tensor.
    :param target: The ground truth class number.
    :param output_class: The predicted class number.
    :param counter: The test image number.
    """
    image = denormalize(tensor).cpu()
    image = image.squeeze(0).permute((1, 2, 0)).numpy()
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = target.cpu().numpy()
    # Enlarge the image a bit to accomodate for the large
    # class names.
    image = cv2.resize(image, (512, 512))
    cv2.putText(
        image, f"GT: {CLASS_NAMES[int(gt)]}",
        (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 255, 0), 2, cv2.LINE_AA
    )
    if output_class == gt:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(
        image, f"Pred: {CLASS_NAMES[int(output_class)]}",
        (5, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, color, 2, cv2.LINE_AA
    )
    cv2.imwrite(
        os.path.join(test_result_save_dir, 'test_image_' + str(counter) + '.png'),
        image * 255.
    )


def test(model, testloader, device, test_result_save_dir):
    """
    Function to test the trained model on the test dataset.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param device: The computation device.
    :param test_result_save_dir: Path to save the resulting images.

    Returns:
        predictions_list: List containing all the predicted class numbers.
        ground_truth_list: List containing all the ground truth class numbers.
        acc: The test accuracy.
    """
    model.eval()
    print('Testing model')
    predictions_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass.
            outputs = model(image)
            # Softmax probabilities.
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            # Predicted class number.
            output_class = np.argmax(predictions)
            # Append the GT and predictions to the respective lists.
            predictions_list.append(output_class)
            ground_truth_list.append(labels.cpu().numpy())
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()
            if i == 200:
                save_test_results(
                    image,
                    labels,
                    output_class,
                    counter,
                    test_result_save_dir
                )

    acc = 100. * (test_running_correct / len(testloader.dataset))
    return predictions_list, ground_truth_list, acc


if __name__ == '__main__':
    weights_path = pathlib.Path(args['weights'])
    model_name = str(weights_path).split(os.path.sep)[-2]
    print(model_name)
    test_result_save_dir = os.path.join('', 'outputs', 'test_results', model_name)
    os.makedirs(test_result_save_dir, exist_ok=True)

    dataset_test = get_datasets(IMAGE_RESIZE)
    test_loader = get_data_loader(dataset_test)

    checkpoint = torch.load(weights_path)
    # Load the model.
    model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictions_list, ground_truth_list, acc = test(
        model,
        test_loader,
        DEVICE,
        test_result_save_dir
    )
    with open(os.path.join(test_result_save_dir, "accuracy.txt"), "w") as acc_file:
        acc_file.write(f"{model_name} Test Accuracy: {acc:.3f}%")
    print(f"Test accuracy: {acc:.3f}%")
