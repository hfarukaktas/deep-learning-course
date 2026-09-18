from pathlib import Path
import torch
import torchvision
from torchvision import transforms
import ModelCreation, setup_data

def predict(image_path: str, model_path: str = 'models/desert_classifier.pth'):
    train_dir = "../data/desert101/train"
    test_dir = "../data/desert101/test"

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    _, _, class_names = setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=32
    )

    loaded_model = ModelCreation.DesertClassifier(
        input_shape=3,
        hidden_units=32,
        output_shape=len(class_names)
    )

    # Modeli yükle
    loaded_model.load_state_dict(torch.load(model_path))

    # Tek görseli işle
    single_image = torchvision.io.read_image(image_path).type(torch.float32) / 255.0
    single_image_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
    ])
    single_image = single_image_transform(single_image).unsqueeze(dim=0)

    loaded_model.eval()
    with torch.inference_mode():
        logits = loaded_model(single_image)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()

    print("Predicted class:", class_names[pred_idx])
    return class_names[pred_idx]

# Sadece bu dosya doğrudan çalıştırıldığında çalışsın:
if __name__ == "__main__":
    predict("data/baklava-online.jpg")