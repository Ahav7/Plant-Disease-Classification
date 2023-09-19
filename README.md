# Plant_Disease_Classification
This project aims to classify plant diseases from the plant village dataset. This repository contains code to train and test the dataset. Also it contains code to run the trained model in a web portal developed in flask.

## Getting started
To get started with the project , follow the steps:
1. Download the dataset from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?resource=download
2. Extract the zipped data and make sure the root folder of the dataset is named "plantvillage dataset"
3. Make sure you have installed all the dependencies like(tqdm, torch, torchinfo, torchvision, matplotlib).
4. Before training, run the command: python prepare_test_data.py
    This will split the dataset into train and test sets in the ratio 80:20. If you want to change the ratio, change the value of "test_split" variable inside the script.
5. Train the model using command: python train.py -e 10 -lr 0.001 -m densenet
   (use -m efficientnet if you want to train with efficientnet)
6. If you want to test, use command: python test.py -w "path/to/best_model.pth"
7. After training, you can directly run the flask app using the command: flask run
8. Open http://127.0.0.1:5000/ to launch the web portal where you can upload your own plant leaf images.

## Dependencies
Make sure you have installed the following dependencies:

* tqdm
* torch
* torchinfo
* torchvision
* matplotlib
* Flask
* opencv-python
* tensorflow=2.0.0
* scikit-learn

You can install these dependencies using `pip`:
```python
pip install tqdm torch torchinfo torchvision matplotlib Flask
```

## Training the Model

To train the model, navigate to the project directory and run the following command in your terminal:

```python
python train.py -e 10 -lr 0.001 -m densenet
```

This will train the model for 10 epochs with a learning rate of 0.001 using the `DenseNet` architecture. If you want to use the `EfficientNet` architecture instead, use the following command:
```python
python train.py -e 10 -lr 0.001 -m efficientnet
```

## Testing the Model
To test the model use the following command
```python
python test.py -w "path/to/best_model.pth"
```
Replace path/to/best_model.pth with the path to your best trained model. This will test the model on the test dataset and print the accuracy.

## Run the Flask
To run the flask use the following command:
```python
flask run
```
This will start the app on http://127.0.0.1:5000/. Open this URL in your web browser to launch the web portal where you can upload your own plant leaf images and get predictions from the model.

## Results
![alt text](https://github.com/lohithreddy15/DLProject/blob/main/Plant%20Disease%20Classification/Diseased%20Grape%20leaf.png)

When an image is uploaded to the interface, the system processes it and provides an output that includes the species of the leaf and the type of disease that it is affected by, we can see the result as grape leaf and it has a blackrot disease.

## Credits
This project was developed by:
* Hemanth Krishna Sai Ram Aradhyula
* Koushik Ponugoti
* Lohith Reddy Nimma


