# BitNetを動かす

[公式のGitHubリポジトリ](https://github.com/microsoft/BitNet)

## 環境構築

0. 要件
    - python>=3.9
    - cmake=>3.22
    - clang>=18

- 自分の環境:
    - python == 3.12.7
    - cmake == 3.30.5
    - clang == 18.1.9

公式はcondaを推奨しているけど、Poetryでもいけるかな？ -> いけそう！

1. リポジトリのclone

```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```

2. 仮想環境の構築

```bash
poetry init
poetry config --local virtualenvs.in-project true
```

3. 依存パッケージのインストール

```bash
poetry run pip install -r requirements.txt
```

4. ビルド

```bash
poetry run python setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s # 結構時間がかかる
```

自分の環境では、HFからGGUFへの変換でコケてる

```txt
$ poetry run python setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s
INFO:root:Compiling the code using CMake.
INFO:root:Downloading model HF1BitLLM/Llama3-8B-1.58-100B-tokens from HuggingFace to models/Llama3-8B-1.58-100B-tokens...
INFO:root:Converting HF model to GGUF format...
ERROR:root:Error occurred while running command: Command '['/home/aki/run_BitNet/BitNet/.venv/bin/python', 'utils/convert-hf-to-gguf-bitnet.py', 'models/Llama3-8B-1.58-100B-tokens', '--outtype', 'f32']' died with <Signals.SIGKILL: 9>., check details in logs/convert_to_f32_gguf.log
```

ログを読んでも原因が分らない！
