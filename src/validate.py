import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from paths import *

def validate(model, device):
    # Set the batch size for validation
    batch_size = 64

    # Define the transform for the validation data
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the validation data
    val_data = datasets.ImageFolder(VALIDATION_DIR, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)

    # Set the model to evaluate mode
    model.eval()

    # Define the criterion for validation
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize the variables for validation
    total_correct = 0
    total_images = 0
    i = 0

    # Validate the model on the validation set
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                 profile_memory=True) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_images += labels.size(0)
                    total_correct += (predicted == labels).sum().item()
                    i += 1

    # Compute the accuracy and print the result
    accuracy = 100 * total_correct / total_images
    print('Accuracy on the validation set: {:.2f}%'.format(accuracy))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return accuracy, str(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))