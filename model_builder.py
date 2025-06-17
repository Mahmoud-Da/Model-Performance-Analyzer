import torch
import torchvision
from torch import nn

# Dictionary to map model names to their creation functions and required input features
# To add a new model, simply add a new entry to this dictionary.
# Format: "model_name": (model_creation_function, weights_enum, in_features_for_classifier)
SUPPORTED_MODELS = {
    "effnetb0": (torchvision.models.efficientnet_b0, torchvision.models.EfficientNet_B0_Weights.DEFAULT, 1280),
    "effnetb2": (torchvision.models.efficientnet_b2, torchvision.models.EfficientNet_B2_Weights.DEFAULT, 1408),
    "resnet18": (torchvision.models.resnet18, torchvision.models.ResNet18_Weights.DEFAULT, 512),
    "resnet50": (torchvision.models.resnet50, torchvision.models.ResNet50_Weights.DEFAULT, 2048),
}


def create_model(model_name: str, num_classes: int, device: str):
    """Creates a model instance from torchvision.models.

    Args:
        model_name (str): The name of the model. Must be a key in SUPPORTED_MODELS.
        num_classes (int): The number of output classes for the model.
        device (str): The target device ("cuda" or "cpu").

    Returns:
        torch.nn.Module: The created model.
    """
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. Please choose from {list(SUPPORTED_MODELS.keys())}")

    model_fn, weights, in_features = SUPPORTED_MODELS[model_name]

    # Create the model with pre-trained weights
    model = model_fn(weights=weights)

    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False

    # Get the name of the classifier layer (it's different for some models)
    if hasattr(model, 'classifier'):
        classifier_layer_name = 'classifier'
    elif hasattr(model, 'fc'):
        classifier_layer_name = 'fc'
    else:
        raise AttributeError(
            f"Model {model_name} has no 'classifier' or 'fc' layer to replace.")

    # Re-create the classifier head to match our number of classes
    setattr(model, classifier_layer_name, nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes)
    ))

    print(f"[INFO] Created new {model_name} model...")
    return model.to(device)
