import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import clip
from PIL import Image
import json
import os

class CombinedMiniWoBDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.data = []
        # データセットファイルが存在するか確認し、存在する場合は読み込む
        if os.path.exists(dataset_path):
            with open(dataset_path, 'r') as f:
                self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        text = item['text']
        return image, text

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

def fine_tune_clip(model, train_loader, optimizer, criterion, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        for images, texts in train_loader:
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)

            optimizer.zero_grad()

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(len(images)).to(device)
            total_loss = (criterion(logits_per_image, ground_truth) + criterion(logits_per_text, ground_truth)) / 2

            total_loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}, Loss: {total_loss.item()}")

def main():
    model, preprocess, device = load_clip_model()

    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
    ])
    
    # 正解データセットと調整された誤答データセットを組み合わせて読み込む
    combined_dataset = CombinedMiniWoBDataset(
        dataset_path='combined_dataset.json',
        transform=transform
    )
    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = F.cross_entropy

    fine_tune_clip(model, train_loader, optimizer, criterion, device, epochs=10)
    
    # モデルの状態辞書を保存するパスを指定
    model_save_path = "./model/finetuned_clip_model.pth"

    # モデルの状態辞書を保存
    torch.save(model.state_dict(), model_save_path)

    print(f"モデルが {model_save_path} に保存されました。")

if __name__ == "__main__":
    main()
