# 親イメージとして公式のPythonランタイムを使用
FROM python:3.9-slim

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# アプリケーションのソースコードをホストからイメージのファイルシステムにコピー
# .dockerignore を使用して不要なファイルを除外
COPY . .
COPY config.json ./config.json
COPY requirements.txt ./requirements.txt

# requirements.txtに記載されている必要なパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install selenium

# Gymライブラリを最新バージョンに更新
RUN pip install --upgrade gym

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y wget gnupg ca-certificates unzip curl

# Googleの公式リポジトリキーを追加
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -

# Google Chromeの公式リポジトリを追加
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" > /etc/apt/sources.list.d/google.list'

# リポジトリキーを明示的に追加
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 4EB27DB2A3B88B8B

# Google Chromeをインストール
# RUN apt-get update && apt-get install -y google-chrome-stable

RUN apt-get update --fix-missing && apt-get install -y --fix-missing google-chrome-stable
RUN google-chrome-stable --version

# # ChromeDriverをダウンロードしてインストール
# RUN wget -q "https://chromedriver.storage.googleapis.com/$(curl -s https://chromedriver.storage.googleapis.com/LATEST_RELEASE)/chromedriver_linux64.zip" -O /tmp/chromedriver.zip
# RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# ChromeDriverをダウンロードしてインストール
RUN wget -q "https://chromedriver.storage.googleapis.com/120.0.6099.109/chromedriver_linux64.zip" -O /tmp/chromedriver.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/

# ChromeDriverとchromeのパスを通す
ENV CHROME_PATH /usr/bin/google-chrome
ENV PATH="/usr/local/bin/chromedriver:${PATH}"

# MiniWoB++をセットアップ
RUN cd computergym && pip install -e .

# コンテナ起動時にbashシェルを開始
CMD ["/bin/bash"]

