import os
import torch
import cv2
import numpy as np
from model import build_model
import torch.nn.functional as F
import torchvision.transforms as transforms
from class_names import class_names as CLASS_NAMES

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Constants and other configurations.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224

weights_path = "outputs/densenet/best_model.pth"
model_name = str(weights_path).split(os.path.sep)[-1]
if not torch.cuda.is_available():
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(weights_path)
# Load the model.
model = build_model(fine_tune=False, num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(model_name)
print(f'Model loaded in device: {DEVICE}. Check http://127.0.0.1:5000/')


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


def infer(model, testloader, DEVICE):
    """
    Function to run inference.

    param model: The trained model.
    param testloader: The test data loader.
    param DEVICE: The computation device.
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


def model_predict(img_path, model):
    transform = get_test_transform(IMAGE_RESIZE)
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    plant_name, disease_class = infer(model, image, DEVICE)
    return plant_name, disease_class


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        print(file_path)
        f.save(file_path)

        # Make prediction
        plant_name, disease_name = model_predict(file_path, model)
        print('Prediction:', plant_name, disease_name)
        result = plant_name + ", " + disease_name
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    model_predict("input/inference_data/corn_common_rust.jpg", model)
    app.run(debug=True)
