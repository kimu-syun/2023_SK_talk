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

# テンプレート読み込み
# with open(R".\parent_act.csv", encoding="utf8") as f:
#     reader = csv.reader(f)
#     act_template = [row for row in reader]

system_message = f"""
# 命令書
あなたは、「子供に部屋の掃除をさせようとする親」です。
以下の制約条件の下で子供への返答を出力してください。
# 制約条件
・子供に対する返答を出力してください。
・回答は返答のみ答えてください。説明は不要です。
・入力される感情は発言者である親の感情です。
・感情はRUSSELLの感情モデルに基づいてください。
・valenceとarousalの値は-10~10の範囲で考えてください。
・1番目がvalence値、2番目がarousal値を表します。
""".strip()


# x(list), text(str)を受け取る
def parent_act(x, text, log):
    if log == []:
        user_res = """
        # 入力(例題)
        ・親の感情 : [2, -6]
        ・子供の発言 : まだ掃除したくないなあ
        """.strip()
        log = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_res},
            {"role": "assistant", "content": "さっさとやってしまいなさい。"},
        ]

    parent_res = f"""
    # 入力
    ・親の感情 : {x}
    ・子供の発言 : {text}
    """.strip()

    user_message = [{"role": "user", "content": parent_res}]
    gpt_msg = log + user_message

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
            parent_talk = res["choices"][0]["message"]["content"].strip()
            print(f"Parent talks to Child : 「{parent_talk}」")
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

    gpt_msg.append(
        {
            "role": "assistant",
            "content": res["choices"][0]["message"]["content"].strip(),
        }
    )

    if res["usage"]["total_tokens"] > 3000:
        gpt_msg.pop(1)
        gpt_msg.pop(1)

    return parent_talk, gpt_msg
