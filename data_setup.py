import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(batch_size: int, num_classes: int, data_name: str):
    """Creates training and testing DataLoaders.

    Uses FakeData for demonstration purposes. Replace with your actual dataset.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names).
    """
    # Create simple transforms
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Using FakeData for a runnable example
    # In a real project, you would use something like:
    # train_data = datasets.ImageFolder(train_dir, transform=data_transform)
    # test_data = datasets.ImageFolder(test_dir, transform=data_transform)

    train_data = datasets.FakeData(size=240,
                                   image_size=(3, 224, 224),
                                   num_classes=num_classes,
                                   transform=data_transform)

    test_data = datasets.FakeData(size=80,
                                  image_size=(3, 224, 224),
                                  num_classes=num_classes,
                                  transform=data_transform)

    class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"[INFO] DataLoader: {data_name}")

    # Turn datasets into DataLoaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader, class_names
