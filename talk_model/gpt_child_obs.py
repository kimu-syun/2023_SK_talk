# 子供が親の発言を感情分析する
import openai
import os
from dotenv import load_dotenv
import sys
import global_value as g
import time

# APIキーをセット
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


# 親の発言textを受け取る
def chi_obs_gpt(text):
    system_message = """
    # 命令書
    あなたは、「親から部屋の掃除をするように言われている子供」です。
    以下の制約条件のもとで入力文に対する感情分析を行い結果を出力してください。
    # 制約条件
    ・入力文は親から子供への発言です。
    ・感情分析はRUSSELLの感情モデルに基づいてください。
    ・valence値とarousal値を-5以上5以下の範囲で答えてください。
    ・1番目にvalence値、2番目にarousal値を入れてください。
    """.strip()

    gpt_msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "部屋の掃除をしなさい！"},
        {"role": "assistant", "content": "-3 3"},
        {"role": "user", "content": text},
    ]

    try_count = 10
    for try_time in range(try_count):
        try:
            res = openai.ChatCompletion.create(
                model=g.gpt_model,
                messages=gpt_msg,
                temperature=0.5,
                max_tokens=10,
                timeout=5,
                request_timeout=5,
            )
            russell_str = res["choices"][0]["message"]["content"].strip().split()

            # ちゃんと出力の形(list[int, int])になっているか確認
            # russell = []
            russell = [int(s) for s in russell_str]
            if (
                len(russell) == 2
                and type(russell[0]) == int
                and type(russell[1]) == int
                and -5 <= russell[0] <= 5
                and -5 <= russell[1] <= 5
            ):
                break
        except ValueError:
            pass
        except (openai.error.APIError, openai.error.Timeout):
            time.sleep(1)
        except openai.error.InvalidRequestError:
            pass
        except (
            openai.error.RateLimitError,
            openai.error.openai.error.APIConnectionError,
        ):
            time.sleep(10)
        print(f"(error {try_time + 1}回目)")

    # return list, list
    return russell


# 𠮟られているかどうかの判定
def chi_scold_gpt(text):
    system_message = """
    # 命令書
    あなたは、「親から部屋の掃除をするように言われている子供」です。
    以下の制約条件のもとで「入力文があなたを叱っているか」を判定してください。
    # 制約条件
    ・入力文は親から子供への発言です。
    ・感情分析はRUSSELLの感情モデルに基づいてください。
    ・叱っていると感じた時は1,叱っていないと感じた時は0を出力してください。
    """.strip()

    gpt_msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "部屋の掃除をしなさい！"},
        {"role": "assistant", "content": "1"},
        {"role": "user", "content": text},
    ]

    try_count = 10
    for try_time in range(try_count):
        try:
            res = openai.ChatCompletion.create(
                model=g.gpt_model,
                messages=gpt_msg,
                temperature=0.5,
                max_tokens=10,
                timeout=5,
                request_timeout=5,
            )
            recog_str = res["choices"][0]["message"]["content"].strip()

            # ちゃんと出力の形(int)になっているか確認
            # russell = []
            recog = int(recog_str)
            if (type(recog)):
                break
        except (openai.error.APIError, openai.error.Timeout):
            time.sleep(1)
        except openai.error.InvalidRequestError:
            pass
        except (
            openai.error.RateLimitError,
            openai.error.openai.error.APIConnectionError,
        ):
            time.sleep(10)
        print(f"(error {try_time + 1}回目)")

    # return list, list
    return recog
