# GPTで短文をプロンプトで頑張って感情分析する
import openai
import os
from dotenv import load_dotenv

# APIキーをセット
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 深津式プロンプト
system_message = """
# 命令書 あなたは、「親から部屋の掃除をするように言われている子供」です。
以下の制約条件のもとで入力文に対する感情分析を行い結果を出力してください。
# 制約条件
・入力文は全て親からあなたへの発言です。
・感情分析はRUSSELLの感情モデルに基づいてください。
・arousalとvalenceの値を-100~100の範囲で答えてください。
・1番目にarousal値、2番目にvalence値を入れてください。
""".strip()

user_message = "部屋の掃除をしてね"

while True:
    try:
        gpt_msg = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "部屋の掃除をしなさい！"},
            {"role": "assistant", "content": "-50 -10"},
            {"role": "user", "content": user_message},
        ]
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=gpt_msg, temperature=2
        )
        print(res["choices"][0]["message"]["content"].strip().split())
        print(type(res["choices"][0]["message"]["content"].strip().split()))
        break
    except:
        print("error")
        break
