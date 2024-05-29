フォルダ内に詳細のmdおいてます。

使用モジュール(pymdp)の公式ドキュメント、チュートリアルのサイトです。

https://pymdp-rtd.readthedocs.io/en/latest/index.html

- `talk_model`
  - 最新版。フォルダ内に詳細のmdあります。まずはここ。
- `old_talk_model`
  - ちょっと古い対話モデル。感情の幅が広かったり、観測信号、隠れ状態の要素が最新版より多かったりします。
- `tokkaken`
  - 1番古いやつ。pymdp、chatGPTを使っていないため、自前でFEの計算をやったり、対話をせず信号だけでやりとりしたりしてます。正直見ても見なくてもいいです。

## はじめに
フォルダ内(`talk_model`or`old_talk_model`)で以下の初期設定を行う。
- APIを記述するファイル(`.env`)を作る
- 作ったファイルに`OPENAI_API_KEY = "実際のAPIキー"`って書く
- `pyenv`環境をインストール(`pyenv install 3.11.4`)
- `poetry`でモジュールをインストール(リポジトリに設定ファイル`pyproject.toml`があるので`poetry install`でOK)
- `poetry run python ファイル名`で実行
