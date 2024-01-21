# 親イメージとして公式のPythonランタイムを使用
FROM python:3.9-slim

# コンテナ内の作業ディレクトリを設定
WORKDIR /usr/src/app

# アプリケーションのソースコードをホストからイメージのファイルシステムにコピー
# .dockerignore を使用して不要なファイルを除外
COPY . .

# requirements.txtに記載されている必要なパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# MiniWoB++をセットアップ
RUN cd computergym && pip install -e .

# コンテナ起動時にbashシェルを開始
CMD ["/bin/bash"]