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
with open(R".\parent_act.csv", encoding="utf8") as f:
    reader = csv.reader(f)
    act_template = [row for row in reader]

system_message = f"""
# 命令書
あなたは、「子供に部屋の掃除をさせようとする親」です。
以下の制約条件の下で入力された発言を感情に適した
子供への返答に変換させて20文字以内で出力してください。
# 制約条件
・子供に対する返答を考えてください。
・あなたの感情と次にとる行動が入力されます。それらに沿った返事をしてください。
・入力される子供の感情はRUSSELLの感情モデルに基づきます。
・valenceとarousalの値は-10~10の範囲で考えてください。
・1番目がvalence値、2番目がarousal値を表します。
・次にとる行動は、[praise(褒める), nutral(何もしない), scold(叱る)]の3種類です。
・20文字以内で回答してください。
・回答は返事のみ答えてください。説明は不要です。
""".strip()


# x(list), a(list)を受け取る
def parent_act(x, a):
    # action抽出
    action = act_template[a[0]][a[1]]

    parent_res = f"""
    # 入力
    ・発言テンプレート：{action}
    ・あなたの感情 : {x}
    ・行動 : {["praise", "nutral", "scold"][a[1]]}
    """.strip()
    user_res = """
    # 入力
    ・発言テンプレート：{"そろそろ掃除しないかな～"}
    ・あなたの感情 : {[2, -6]}
    ・行動 : nutral
    """.strip()
    gpt_msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_res},
        {"role": "assistant", "content": "そろそろ掃除したら？"},
        {"role": "user", "content": parent_res},
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

    return parent_talk
