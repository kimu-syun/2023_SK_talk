## pyenvコマンド
- `pyenv install 3.11.4` インストール
- `pyenv versions` インストール済みのバージョン一覧
- `pyenv locall 3.11.4` バージョン指定

## poetryコマンド
- `poetry new <project name>` スターターセット作成
- `poetry init` pyproject.tomlのみ作成
- `poetry install` pyproject.tomlの環境をインストール
- `poetry add <package name>` パッケージ追加
- `poetry remove <package name>` パッケージ削除
- `poetry run <comand>` poetry下でコマンド実行
- `poetry shell` 仮想poetry環境内でのシェルを立ち上げる


## 使い方
親子の対話シミュレーションを行う。設定は、親が子供に掃除をするように言い子供がそれに対して応答する。
`gpt_talk.py`が対話シミュレーションを行うファイル。これを実行する。

入力
- 対話を何回行うか
- 親の感情はなにか
- 期待する観測信号はなにか
出力
- 対話ログ（親と子の対話）
- グラフ
  - 隠れ状態の推定分布と真の親の感情
  - 選好分布と受け取った観測信号


### 変数
- 隠れ状態 : [parent emotion(21 × 21)]
- 観測信号 : [parent emotion(21 × 21)]
- 行動 : [child act(3)]


### 生成モデル
- A : P(o|s)
  - 1回分の応答をChatGPTに生成させた。各隠れ状態ごと100回分の結果をもとに生成。（`likelihood.py`）
- B : P(s|s,a)
- C : P(o)
- D : P(s)


## 環境設定
ChatGPT : 4.0の方がプロンプトの内容がしっかり反映される。(GPTは`gpt_child_act.py`, `gpt_child_obs.py`, `gpt_parent_act.py`で使用される。)


隠れ状態 : ラッセルの円環モデルで定義。arousal, valenceをそれぞれ-5:5で定義。(11×11)
観測信号 : ラッセルの円環モデルで定義。arousal, valenceをそれぞれ-5:5で定義。(11×11)  親の発言を感情識別にかける。
行動 : 子供の行動。対応行動を5種類あらかじめ作成した。


## 今後の課題
- 今回は感情は変化しないため遷移分布がすべて1であったが、行動に差をつけるために感情を遷移した状態で行う。
- 子供の感情を観測信号にすることで、親の感情と自分の感情のバランスをとった行動がとれることが期待される、かも。
- 学習モジュールを使用することができなかったため、これを使ってやりたい。