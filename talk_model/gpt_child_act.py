# 子供の発言を決定する
import openai
import os
from dotenv import load_dotenv
import sys
import csv
import random
import global_value as g
import time

# APIキーをセット
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

system_message = f"""
# 命令書
あなたは、「親に掃除するように言われている子供」です。
以下の制約条件の下で返答を出力してください。
# 制約条件
・親の発言に対する返答を考えてください。
・「親の発言」と「子供の対応行動」が入力されます。これをもとに自然な返答をしてください。
・返答のみ答えてください。説明は不要です。
""".strip()


# action, par_textを受け取る
def chi_text_gpt(act, text):

    child_res = f"""
    # 入力
    ・親の発言 : {text}
    ・子供の対応行動 : {act}
    """.strip()
    gpt_msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": child_res},
    ]

    try_count = 10
    for try_time in range(try_count):
        try:
            res = openai.ChatCompletion.create(
                model=g.gpt_model,
                messages=gpt_msg,
                temperature=1,
                max_tokens=50,
                timeout=5,
                request_timeout=5,
            )
            child_talk = res["choices"][0]["message"]["content"].strip()
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

    return child_talk
