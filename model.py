from torchvision import models
import torch.nn as nn


def model_config(model_name='efficientnet'):
    model = {
        'densenet': models.densenet121(pretrained=True),
        'efficientnet': models.efficientnet_b0(weights='DEFAULT')
    }
    return model[model_name]


def build_model(model_name='efficientnet', fine_tune=True, num_classes=10):
    model = model_config(model_name)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    if model_name == 'densenet':
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    if model_name == 'efficientnet':
        model.classifier[1].out_features = num_classes
    return model
