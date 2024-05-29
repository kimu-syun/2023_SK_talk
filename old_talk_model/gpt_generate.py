# GPTでテンプレートを感情をもとに言い換える
import openai
import os
from dotenv import load_dotenv

# APIキーをセット
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
