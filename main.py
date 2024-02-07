import argparse
import random

import computergym
import gym
from llm_agent import LLMAgent

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

from bs4 import BeautifulSoup

from tqdm import tqdm

import logging

import os
import datetime
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')
now = datetime.datetime.now(JST)

logging.basicConfig(level=logging.INFO)

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

    opt = parser.parse_args()

    return opt

# スクリーンショットの格納先フォルダを作成 -> clip_dataset
def setup_dataset_directory(base_dir="clip_dataset"):
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    # csv_file_path = os.path.join(base_dir, "dataset.csv")
    '''
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["image_filename", "text_description", "button_texts"])
    '''
    # return images_dir, csv_file_path
    return images_dir

# HTMLのbutton要素を取得
def get_button_list(html_state):
    button_list = []
    soup = BeautifulSoup(html_state, "html.parser")
    button_id = "count-buttons"
    button_elements = soup.find_all("button")
    for button in button_elements:
        button_list.append(int(button.text)) 
    return button_list

# 英数字、図形部分のHTMLの画像をスクリーンショットで撮る
def get_screenshot(html_state):
    
    ### HTMLを指定の格納先に保存
    
    # HTMLの格納先を指定
    current_dir = os.getcwd()
    file_path = './clip/miniwob/html/count-shape.html'
    # ファイルを書き込みモードで開き、HTMLコンテンツを書き込む
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(html_state)
    ### スクリーンショットを指定の格納先に保存

    # ファイルをChromeで開く
    file_path_url = 'file:///' + os.path.abspath(file_path)
    print("file_path_url is" + file_path_url)
    driver = get_webdriver(file_path_url)
    svg_element = driver.find_element(By.ID, "area_svg")
    image_filename = f"image_{now:%Y%m%d%H%M%S}.png"
    images_dir = './clip/past_data/clip_dataset/images'
    image_path = os.path.join(images_dir, image_filename)
    # スクリーンショットを取得
    svg_element.screenshot(image_path)
    driver.quit()
    # HTMLファイルを削除
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

# 【!要確認!】miniwob環境のときは以下の関数を実行
def miniwob(opt):
    env = gym.make("MiniWoBEnv-v0", env_name=opt.env, headless=opt.headless)

    success = 0
    for _ in tqdm(range(opt.num_episodes)):
        llm_agent = LLMAgent(
            opt.env,
            rci_plan_loop=opt.erci,
            rci_limit=opt.irci,
            llm=opt.llm,
            state_grounding=opt.sgrounding,
        )
        # initialize environment
        states = env.reset(seeds=[random.random()], record_screenshots=True)
        llm_agent.set_goal(states[0].utterance)
        html_state = get_html_state(opt, states)

        # HTMLを出力するテスト 
        # print(html_state)
        llm_agent.update_html_state(html_state)

        try:
            llm_agent.initialize_plan()
        except:
            continue

        if opt.step == -1:
            step = llm_agent.get_plan_step()
        else:
            step = opt.step

        logging.info(f"The number of generated action steps: {step}")

        for _ in range(step):
            assert len(states) == 1
            try:
                instruction = llm_agent.generate_action()
                logging.info(f"The executed instruction: {instruction}")

                miniwob_action = llm_agent.convert_to_miniwob_action(instruction)

                states, rewards, dones, _ = env.step([miniwob_action])
            except ValueError:
                print("Invalid action or rci action fail")
                rewards = [0]
                dones = [True]
                break

            if rewards[0] != 0:
                break

            if all(dones):  # or llm_agent.check_finish_plan():
                break

            html_state = get_html_state(opt, states)
            llm_agent.update_html_state(html_state)

        if rewards[0] > 0:
            success += 1
            llm_agent.save_result(True)
        else:
            llm_agent.save_result(False)

        print(f"success rate: {success / opt.num_episodes}")

    env.close()

# 【!要確認!】count-shape環境のときは以下の関数を実行
def miniwob_count_shape(opt):
    # 挙動テスト用print
    # print(count_shape_str)

    # ライブラリgymでMiniWobの実行環境をインスタンス化
    env = gym.make("MiniWoBEnv-v0", env_name=opt.env, headless=opt.headless)

    success = 0
    for _ in tqdm(range(opt.num_episodes)):
        # llm_agent.pyに指定のタスク種別(count-shape)等の情報を渡している
        llm_agent = LLMAgent(
            opt.env,
            rci_plan_loop=opt.erci,
            rci_limit=opt.irci,
            llm=opt.llm,
            state_grounding=opt.sgrounding,
        )

        # initialize environment
        # ライブラリgymでreset関数により、gymの実行環境を初期化
        states = env.reset(seeds=[random.random()], record_screenshots=True)

        # テスト用コード：count-shapeの指示文を出力
        # print("Our goal is" + states[0].utterance)

        # llm_agent内でcount-shapeの問題文を利用できるようにしている
        llm_agent.set_goal(states[0].utterance)

        # count-shapeタスクのhtml情報を取得
        html_state = get_html_state(opt, states)

        # llm_agent.pyにhtmlを渡す処理
        llm_agent.update_html_state(html_state)

        ## beautifulsoupで選択肢リストを取得する
        button_list = get_button_list(html_state)
        # print(button_list)
        ## seleniumでスクリーンショットを撮り、画像を保存する
        image_path = get_screenshot(html_state)
        ## CLIPで画像を読み込み、最も確率の高い選択肢を抽出する
        # button_num = get_button_num_from_clip(states[0].utterance,button_list)
        button_num = 1
        # クリックするボタンに最も確率の高い選択肢を適用する
        instruction = f"clickxpath //*[@id=\"count-buttons\"]/button[{button_num}]"
        try:
            # instruction = llm_agent.generate_action()
            # print("instruction is" + instruction)
            logging.info(f"The executed instruction: {instruction}")
            
            miniwob_action = llm_agent.convert_to_miniwob_action(instruction)

            # envに結果を返してリワードを更新する
            states, rewards, dones, _ = env.step([miniwob_action])
        except ValueError:
            print("Invalid action or rci action fail")
            rewards = [0]
            dones = [True]

        # 成功していれば、rewardは正になる
        if rewards[0] > 0:
            success += 1
            llm_agent.save_result(True)
        else:
            llm_agent.save_result(False)

        print(f"success rate: {success / opt.num_episodes}")


    env.close()


def get_html_state(opt, states):
    extra_html_task = [
        "click-dialog",
        "click-dialog-2",
        "use-autocomplete",
        "choose-date",
    ]

    html_body = states[0].html_body
    if opt.env in extra_html_task:
        html_body += states[0].html_extra
    return html_body

# メインの実行タスク
if __name__ == "__main__":
    # 入力したタスク種別の情報を取得
    opt = parse_opt()
    if opt.env == "facebook":
        url = "https://www.facebook.com/"
        web(opt, url)
    elif opt.env == "count-shape":
        miniwob_count_shape(opt)
    else:
        miniwob(opt)
