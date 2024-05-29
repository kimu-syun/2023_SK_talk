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


# text, par_emoを受け取る
def par_text_gpt(text=None, par_emo=None, first=False):

    if first == True:
        system_message = f"""
            # 命令書
            あなたは、「子供に部屋の掃除をさせたい親」です。
            以下の制約条件の下で呼びかけを出力してください。
            # 制約条件
            ・子供への呼びかけを考えてください。
            ・入力された親の感情を強く反映させて呼びかけをしてください。
            ・入力される親の感情はRUSSELLの感情モデルに基づきます。
            ・1番目がvalence値、2番目がarousal値を表します。
            ・valence値とarousal値は-5~5の範囲で考えてください。
            ・呼びかけのみ答えてください。説明は不要です。
            """.strip()
        parent_res = f"""
        # 入力
        ・親の感情 : {par_emo}
        """.strip()
        gpt_msg = [
            {"role": "system", "content": system_message},
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

    else:
        system_message = f"""
            # 命令書
            あなたは、「子供に部屋の掃除をさせたい親」です。
            以下の制約条件の下で返答を出力してください。
            # 制約条件
            ・子供に対する返答を考えてください。
            ・入力された親の感情に従って返事をしてください。
            ・入力される親の感情はRUSSELLの感情モデルに基づきます。
            ・1番目がvalence値、2番目がarousal値を表します。
            ・valence値とarousal値は-5~5の範囲で考えてください。
            ・返答のみ答えてください。説明は不要です。
            """.strip()

        parent_res = f"""
        # 入力
        ・子供の発言：{text}
        ・親の感情 : {par_emo}
        """.strip()
        gpt_msg = [
            {"role": "system", "content": system_message},
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
