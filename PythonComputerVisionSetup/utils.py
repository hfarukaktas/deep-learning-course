import torch
from torchvision import transforms
from pathlib import Path

def save_model(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str,
):
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    model_save_path = target_dir_path / model_name

    torch.save(obj = model.state_dict(), f = model_save_path)

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_images_count = 0
    for images, _ in loader:
        # Resim sayısı (batch_size, channels, height, width)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count
    return mean, std

if __name__ == '__main__':
    import setup_data
    NUM_EPOCHS = 10
    BATCH_SIZE = 32
    HIDDEN_UNITS = 32
    LEARNING_RATE = 0.001

    train_dir = "../data/desert101/train"
    test_dir = "../data/desert101/test"

    data_transform = transforms.Compose(
        [
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ]
    )

    train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )
    mean, std = get_mean_and_std(train_dataloader)
    print(f"Mean: {mean}")
    print(f"Std: {std}")