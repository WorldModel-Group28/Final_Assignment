import os
import csv
import clip
import torch
from PIL import Image

class CLIP:
    def load_clip_model():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, device

    def predict_choice(model, preprocess, device, image_path, text_descriptions):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        text = clip.tokenize(text_descriptions).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            # 類似度スコアの計算（修正部分）
            logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
            probs = logits_per_image.cpu().numpy()

        return probs[0]


    def get_button_num(str, button_list):
        print("Hello CLIP Hello CLIP Hello CLIP")
        return 0
        '''
        model, preprocess, device = load_clip_model()
        base_dir = "clip_dataset"
        images_dir = os.path.join(base_dir, "images")
        csv_file_path = os.path.join(base_dir, "dataset.csv")

        with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_path = os.path.join(images_dir, row["image_filename"])
                text_description = row["text_description"]
                button_texts = row["button_texts"].split('|')
                text_descriptions = [f"{text_description} {btn_text}" for btn_text in button_texts]

                probs = predict_choice(model, preprocess, device, image_path, text_descriptions)
                best_choice_index = probs.argmax()
                print(f"Image: {row['image_filename']}, Best choice: {button_texts[best_choice_index]}, Probability: {probs[best_choice_index]:.4f}")
        '''        