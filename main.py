import argparse
import random

import computergym
import gym
from llm_agent import LLMAgent

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains

from tqdm import tqdm

import logging

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

    driver = webdriver.Chrome(chrome_options=options)
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

        # プロンプトにgoalとHTMLを渡して解かせる
        try:
            instruction = llm_agent.generate_action()
            logging.info(f"The executed instruction: {instruction}")

            miniwob_action = llm_agent.convert_to_miniwob_action(instruction)

            states, rewards, dones, _ = env.step([miniwob_action])
        except ValueError:
            print("Invalid action or rci action fail")
            rewards = [0]
            dones = [True]

        # テスト用コード：HTMLがちゃんと取得できているか出力
        # print(html_state)

        ''' 一度llm_agentから出力するだけなので、step数をカウントすることも不要、initializeのプロンプトも不要
        # タスク指示とHTMLをプロンプトに渡して、プロンプトからの返答内容をptに保存する
        try:
            llm_agent.initialize_plan_count_shape()
        except:
            continue
        
        # 実行するプラン数をカウントする
        if opt.step == -1:
            step = llm_agent.get_plan_step()
        else:
            step = opt.step

        logging.info(f"The number of generated action steps: {step}")

        #### プランの数だけ下記処理を実行する→ここを変更する
        # 1.プロンプトから該当の答えを持ってくる
        # 2.回答されたものを""等で区切っておき、それだけの文字列にする
        # 3.buttonのelementを取得してクリックする
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
        '''

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
