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


# log : 会話の履歴(list[{}])
# parent_msg : 親の発言(str)
def child_obs(log, parent_msg, parent_action):
    # 初期値設定（会話の例）
    if log == []:
        # 深津式プロンプト
        system_message = """
        # 命令書
        あなたは、「親から部屋の掃除をするように言われている子供」です。
        以下の制約条件のもとで入力文に対する感情分析を行い結果を出力してください。
        # 制約条件
        ・入力文は全て親からあなたへの発言です。
        ・親の発言は[praise(褒める), nutral（普通）, scold(叱る)]の3種類に分類されます。
        ・感情分析はRUSSELLの感情モデルに基づいてください。
        ・valenceとarousalの値を-10~10の範囲で答えてください。
        ・1番目にvalence値、2番目にarousal値を入れてください。
        """.strip()
        log = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "部屋の掃除をしなさい！ (scold)"},
            {"role": "assistant", "content": "-7 6"},
        ]

    # メッセージ設定
    user_message = [{"role": "user", "content": f"{parent_msg} ({parent_action})"}]
    gpt_msg = log + user_message


    try_count = 10
    for try_time in range(try_count):
        try:
            res = openai.ChatCompletion.create(
                model=g.gpt_model, messages=gpt_msg, temperature=0.5, max_tokens = 10,
                timeout=5, request_timeout=5
            )
            russell_str = res["choices"][0]["message"]["content"].strip().split()

            # ちゃんと出力の形(list[int, int])になっているか確認
            # russell = []
            russell = [int(s) for s in russell_str]
            if (
                len(russell) == 2
                and type(russell[0]) == int
                and type(russell[1]) == int
            ):
                print(f"Child observation : {russell}")
                break
        except (openai.error.APIError,
                openai.error.Timeout):
            time.sleep(1)
        except openai.error.InvalidRequestError:
            pass
        except (openai.error.RateLimitError,
                openai.error.openai.error.APIConnectionError):
            time.sleep(10)
        print(f"(error {try_time + 1}回目)")



    # 最新の答えを追加
    gpt_msg.append(
        {
            "role": "assistant",
            "content": res["choices"][0]["message"]["content"].strip(),
        }
    )

    # トークンが多い場合古いものを削除
    if res["usage"]["total_tokens"] > 3000:
        gpt_msg.pop(1)
        gpt_msg.pop(1)

    # return list, list
    return russell, gpt_msg