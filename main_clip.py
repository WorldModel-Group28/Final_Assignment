import argparse
import random
import re
import gym
from llm_agent import LLMAgent
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
import os
import datetime
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import numpy
import glob



logging.basicConfig(level=logging.INFO)
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="click-button")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--llm", type=str, default="chatgpt")
    parser.add_argument("--erci", type=int, default=0)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--irci", type=int, default=1)
    parser.add_argument("--sgrounding", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()

def get_webdriver(file_path_url):
    chrome_options = Options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_screenshot(html_state):
    now = datetime.datetime.now(JST)
    current_dir = os.getcwd()
    file_path = './clip/miniwob/html/count-shape.html'
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(html_state)
    file_path_url = 'file:///' + os.path.abspath(file_path)
    driver = get_webdriver(file_path_url)
    svg_element = driver.find_element(By.ID, "area_svg")
    image_filename = f"image_{now:%Y%m%d%H%M%S}.png"
    images_dir = './clip/past_data/clip_dataset/images'
    image_path = os.path.join(images_dir, image_filename)
    svg_element.screenshot(image_path)
    driver.quit()
    os.remove(file_path)
    return image_path


# 【確認不要】実行環境がfacebookのときだけ実行する
def web(opt, url):
    driver = get_webdriver(url)

    while True:
        llm_agent = LLMAgent(
            opt.env, rci_plan_loop=opt.erci, rci_limit=opt.irci, llm=opt.llm
        )

        html_body = get_html_state_from_real(driver, opt)

        llm_agent.update_html_state(html_body)

        # Set objective (e.g., login with id and pw)
        goal = input("Type your command (type 'exit' to quit): ")
        if goal == "exit":
            break
        llm_agent.set_goal(goal)

        llm_agent.initialize_plan()

        step = llm_agent.get_plan_step()
        logging.info(f"The number of generated action steps: {step}")
        for _ in range(step):
            instruction = llm_agent.generate_action()
            print(instruction)

            perform_instruction(driver, instruction)

            html_body = get_html_state_from_real(driver, opt)
            llm_agent.update_html_state(html_body)

    driver.quit()

# 【不要】実行環境がfacebookのときだけ実行する
def get_html_state_from_real(driver, opt):
    if opt.env == "facebook":
        main_html_xpath = '//*[@id="content"]'
        html_body = driver.find_element(By.XPATH, main_html_xpath).get_attribute(
            "outerHTML"
        )
    else:
        raise NotImplemented

    return html_body

# 【不要】実行環境がfacebookのときだけ実行する
def perform_instruction(driver, instruction):
    instruction = instruction.split(" ")
    inst_type = instruction[0]
    inst_type = inst_type.lower()

    if inst_type == "type":
        characters = " ".join(instruction[1:])
        characters = characters.replace('"', "")
        chain = ActionChains(driver)
        chain.send_keys(characters)
        chain.perform()
    elif inst_type == "clickxpath":
        xpath = " ".join(instruction[1:])
        element = driver.find_element(By.XPATH, str(xpath))
        chain = ActionChains(driver)
        chain.move_to_element(element).click().perform()
    elif inst_type == "press":
        key_type = instruction[1]
        # TODO: press special key
        if key_type == "enter":
            chain = ActionChains(driver)
            chain.send_keys("\n")
            chain.perform()
        elif key_type == "space":
            chain = ActionChains(driver)
            chain.send_keys(" ")
            chain.perform()
        else:
            raise NotImplemented
    else:
        raise ValueError("Invalid instruction")

# 【不要】実行環境がfacebookのときだけ実行する
def get_webdriver(url):
    options = webdriver.ChromeOptions()
    # options.add_argument("headless")
    options.add_argument("disable-gpu")
    options.add_argument("no-sandbox")

    driver = webdriver.Chrome(options=options)
    driver.implicitly_wait(5)
    driver.maximize_window()
    driver.implicitly_wait(5)

    driver.get(url)
    driver.implicitly_wait(10)
    return driver

def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device
    

def predict_choice(model, processor, device, image_path, text_descriptions):
    # 画像を読み込む
    image = Image.open(image_path)
    # 画像とテキストをプロセッサで処理
    inputs = processor(text=text_descriptions, images=image, return_tensors="pt", padding=True)
    
    # 入力データをモデルと同じデバイスに移動
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # これは画像ごとのテキストの類似度
        probs = logits_per_image.softmax(dim=1).cpu().numpy()  # 結果をCPUに移動してからNumpy配列に変換

    return probs[0]


# def predict_choice(model, preprocess, device, image_path, text_descriptions):
#     image = Image.open(image_path)
#     inputs = preprocess(text=text_descriptions, images=image, return_tensors="pt", padding=True).to(device)
#     outputs = model(**inputs)
#     probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
#     return probs[0]

def get_button_num(problem, button_list, image_path):
    model, processor, device = load_clip_model()
    match = re.search(r"How many (.+?) are", problem)
    result = match.group(1) if match else "items"
    text_descriptions = [f"{btn} {result}" for btn in button_list]
    probs = predict_choice(model, processor, device, image_path, text_descriptions)
    for i, text in enumerate(text_descriptions):
        print(f"説明: {text} (類似度: {probs[i]:.4f})")
    top_choice_index = probs.argmax()
    return button_list[top_choice_index], text_descriptions[top_choice_index], probs[top_choice_index]



def convert_float32(o):
    if isinstance(o, numpy.float32):
        return float(o)  # numpy.float32 を float に変換
    elif isinstance(o, numpy.ndarray):
        return o.tolist()  # numpy 配列をリストに変換
    raise TypeError("Object of type '%s' is not JSON serializable" % type(o).__name__)



# def adjust_incorrect_similarity_scores(incorrect_datasets, correct_datasets):
#     if not correct_datasets:
#         return
    
#     # 正解データセットから最大の類似度スコアを見つける
#     max_similarity = max(entry['similarity'] for entry in correct_datasets)

#     # 誤答データセットの類似度スコアを正解データセットの最大類似度に調整
#     for entry in incorrect_datasets:
#         entry['similarity'] = max_similarity

#     return incorrect_datasets

# 以下の誤答問題の類似度スコアを正解番号が最も高くなるように処理する部分が問題

def adjust_incorrect_similarity_scores(incorrect_datasets, correct_datasets, correct_answers):
    # 正答に対する類似度をマッピングする辞書を準備
    correct_similarity_map = {}
    for entry in correct_datasets:
        question_id = entry['count_buttun']  # 正答情報を識別するためのキー
        # 正答の類似度を保存
        correct_similarity_map[question_id] = entry['similarity']

    # 誤答データセット内の各エントリを更新
    for entry in incorrect_datasets:
        question_id = entry['count_buttun']  # 誤答エントリの識別子
        if question_id in correct_answers:
            # 本来の正答番号を取得
            correct_button_num = correct_answers[question_id]
            # 本来の正答番号に対応する類似度が最も高いものに調整
            entry['similarity'] = correct_similarity_map.get(correct_button_num, entry['similarity'])
        else:
            # 正答情報が見つからない場合は、調整しない
            pass

    return incorrect_datasets







def save_dataset(dataset, dataset_type="correct"):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_path = os.path.join(f"{dataset_type}_datasets_{now}.json")
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=4, default=convert_float32)

def combine_and_save_datasets(correct_datasets, incorrect_datasets, combined_file_name="combined_dataset.json"):
    combined_data = []
    
    # combined_dataset.json ファイルが既に存在する場合は、その内容を読み込む
    if os.path.exists(combined_file_name):
        with open(combined_file_name, 'r') as f:
            combined_data = json.load(f)
    
    # 誤答データセットの類似度スコアを調整
    adjusted_incorrect_datasets = adjust_incorrect_similarity_scores(incorrect_datasets, correct_datasets)
    
    # 新たなデータセットを追加
    combined_data.extend(correct_datasets + adjusted_incorrect_datasets)
    
    # 結合したデータセットをファイルに保存
    with open(combined_file_name, 'w') as f:
        json.dump(combined_data, f, indent=4, default=convert_float32)
        
    print(f"Combined dataset saved to {combined_file_name})")


def miniwob_count_shape(opt):
    env = gym.make("MiniWoBEnv-v0", env_name=opt.env, headless=opt.headless)
    success = 0
    correct_datasets = []
    incorrect_datasets = []  # 誤答データを収集するためのリストを追加
    for _ in tqdm(range(opt.num_episodes)):
        states = env.reset(seeds=[random.random()], record_screenshots=True)
        llm_agent = LLMAgent(opt.env, rci_plan_loop=opt.erci, rci_limit=opt.irci, llm=opt.llm, state_grounding=opt.sgrounding)
        llm_agent.set_goal(states[0].utterance)
        html_state = get_html_state(opt, states)
        llm_agent.update_html_state(html_state)
        button_list = get_button_list(html_state)
        image_path = get_screenshot(html_state)
        button_num, correct_text, correct_similarity = get_button_num(states[0].utterance, button_list, image_path)
        dataset_entry = {
            "image_path": image_path,
            "text": correct_text,
            "similarity": correct_similarity,
            "success_rate": success / opt.num_episodes if opt.num_episodes > 0 else 0
        }
        instruction = f"clickxpath //*[@id=\"count-buttons\"]/button[{button_num}]"
        try:
            logging.info(f"The executed instruction: {instruction}")
            miniwob_action = llm_agent.convert_to_miniwob_action(instruction)
            states, rewards, dones, _ = env.step([miniwob_action])
            if rewards[0] > 0:
                success += 1
                llm_agent.save_result(True)
                correct_datasets.append(dataset_entry)  # 成功したら正解データセットに追加
            else:
                llm_agent.save_result(False)
                incorrect_datasets.append(dataset_entry)  # 失敗したら誤答データセットに追加
        except ValueError:
            print("Invalid action or rci action fail")

    # 誤答データセットの分析と類似度スコアの調整（ここでは仮の処理を示します）
    # 実際には、誤答データと正解データの類似度を比較し、適切に調整する必要があります

    print(f"success rate: {success / opt.num_episodes}")


    # データセットの保存
    save_dataset(correct_datasets, "correct")
    save_dataset(incorrect_datasets, "incorrect")
    
        # 主要な処理は以前と同じですが、データセットの保存方法を変更します
    adjusted_incorrect_datasets = adjust_incorrect_similarity_scores(incorrect_datasets, correct_datasets, correct_answers)
    
    # 正解データセットと調整された誤答データセットを結合して保存
    combine_and_save_datasets(correct_datasets, adjusted_incorrect_datasets)


    
 
def get_html_state(opt, states):
    html_body = states[0].html_body
    return html_body

def get_button_list(html_state):
    soup = BeautifulSoup(html_state, "html.parser")
    button_elements = soup.find_all("button")
    button_list = [int(button.text) for button in button_elements if button.text.isdigit()]
    return button_list

if __name__ == "__main__":
    opt = parse_opt()
    if opt.env == "count-shape":
        miniwob_count_shape(opt)
    else:
        miniwob(opt)
