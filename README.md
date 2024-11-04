# BitNetを動かす

[公式のGitHubリポジトリ](https://github.com/microsoft/BitNet)

## 環境構築&ビルド

0. 要件
    - python>=3.9
    - cmake=>3.22
    - clang>=18

- 自分の環境
    - python == 3.12.7
    - cmake == 3.30.5
    - clang == 18.1.9

- 自分の環境(Desktopマシン)
    - OS: Arch Linux x86\_64
    - Kernel: 6.11.4-arch2-1
    - CPU: 13th Gen Intel i7-13700 (24) @ 5.100GHz
    - Memory: 31851MiB

公式はcondaを推奨しているけど、Poetryでもいけるかな？ -> ~~いけそう！~~ いけた

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

自分の環境では、HFからGGUFへの変換でコケてる。

```txt
$ poetry run python setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s
INFO:root:Compiling the code using CMake.
INFO:root:Downloading model HF1BitLLM/Llama3-8B-1.58-100B-tokens from HuggingFace to models/Llama3-8B-1.58-100B-tokens...
INFO:root:Converting HF model to GGUF format...
ERROR:root:Error occurred while running command: Command '['/home/aki/run_BitNet/BitNet/.venv/bin/python', 'utils/convert-hf-to-gguf-bitnet.py', 'models/Llama3-8B-1.58-100B-tokens', '--outtype', 'f32']' died with <Signals.SIGKILL: 9>., check details in logs/convert_to_f32_gguf.log
```

~~ログを読んでも原因が分らない~~

同じような症状は[Issues:  Died with Signals.SIGKILL: 9 when running setup_env.py #27](https://github.com/microsoft/BitNet/issues/27)にもあった。


デフォルトの`HF1BitLLM/Llama3-8B-1.58-100B-tokens`は結局モデルがかなり大規模らしく、モデル変換でメモリが足りなくなって、死んでしまうらしい。
そこでより、小さなモデルである3Bだとそこまでメモリがリッチじゃないマシンでも動くらしいのでそっちを動かしてみる。

```bash
poetry run python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-3B -q i2_s
```

こっちではビルドに成功した。

## モデルを動かす!

```bash
python run_inference.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf -p "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:" -n 6 -temp 0
```

一応動いた、けどanswerが空欄で返答がない(抜粋 [詳細](./detail/detail_1))。

```txt
 Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?
Answer:.

  (

llama_perf_sampler_print:    sampling time =       0.08 ms /    60 runs   (    0.00 ms per token, 789473.68 tokens per second)
llama_perf_context_print:        load time =     250.90 ms
llama_perf_context_print: prompt eval time =    3415.84 ms /    54 tokens (   63.26 ms per token,    15.81 tokens per second)
llama_perf_context_print:        eval time =     322.18 ms /     5 runs   (   64.44 ms per token,    15.52 tokens per second)
llama_perf_context_print:       total time =    3738.71 ms /    59 tokens
```

### 他のモデルも動かす

以下のモデルが使えるらしい(`1bitLLM/bitnet_b1_58-3B`,`HF1BitLLM/Llama3-8B-1.58-100B-tokens` はもう試した。)

- model
    - `1bitLLM/bitnet_b1_58-large`
    - `1bitLLM/bitnet_b1_58-3B`
    - `HF1BitLLM/Llama3-8B-1.58-100B-tokens`

#### `1bitLLM/bitnet_b1_58-large`をビルド&実行する

こっちのモデルもビルドに成功した。
```bash
$ poetry run python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large -q i2_s
INFO:root:Compiling the code using CMake.
INFO:root:Downloading model 1bitLLM/bitnet_b1_58-large from HuggingFace to models/bitnet_b1_58-large...
INFO:root:Converting HF model to GGUF format...
INFO:root:GGUF model saved at models/bitnet_b1_58-large/ggml-model-i2_s.gguf
```

こちらのモデルも動かした。

(一部抜粋 [詳細](./detail/))
```txt
 Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?
Answer: Mary went to the kitchen.

llama_perf_sampler_print:    sampling time =       0.08 ms /    60 runs   (    0.00 ms per token, 731707.32 tokens per second)
llama_perf_context_print:        load time =     118.12 ms
llama_perf_context_print: prompt eval time =    1205.08 ms /    54 tokens (   22.32 ms per token,    44.81 tokens per second)
llama_perf_context_print:        eval time =     116.27 ms /     5 runs   (   23.25 ms per token,    43.00 tokens per second)
llama_perf_context_print:       total time =    1321.96 ms /    59 tokens
```

こっちだと、それっぽい答が出力されていることが分る。


## Laptopの環境でもビルド&実行をしてみる

- laptopマシン
    - OS: Arch Linux x86_64
    - Kernel: 6.11.4-arch1-1
    - CPU: 13th Gen Intel i5-13500H (16) @ 4.700GHz
    - Memory: 15655MiB

では、`1bitLLM/bitnet_b1_58-large` でしかモデル変換に成功しなかった(変換に成功したモデルはちゃんと実行できた)。

## メモリ不足問題の解決策

ビルドで転けた際に参照した[Issues: Died with Signals.SIGKILL: 9 when running setup_env.py #27 ](https://github.com/microsoft/BitNet/issues/27)を読むとどうやらそこまでRAMが多くなくても、
swapがある程度あれば`HF1BitLLM/Llama3-8B-1.58-100B-tokens`の変換に成功するらしい(その人は20GiBのRAMで、20GiBのswap)。
そこで、swapを作成して変換処理を行ってみることにした。

まず、Laptopで20GiBのswap fileを作成してから`1bitLLM/bitnet_b1_58-3B`の変換処理を行なうと成功し、そのモデルの実行にも成功した。
しかし、`HF1BitLLM/Llama3-8B-1.58-100B-tokens`では、変換には成功しなかった。

```bash
 $ poetry run python setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s
INFO:root:Compiling the code using CMake.
INFO:root:Downloading model HF1BitLLM/Llama3-8B-1.58-100B-tokens from HuggingFace to models/Llama3-8B-1.58-100B-tokens...
INFO:root:Converting HF model to GGUF format...
ERROR:root:Error occurred while running command: Command '['/home/aki/run_BitNet/BitNet/.venv/bin/python', 'utils/convert-hf-to-gguf-bitnet.py', 'models/Llama3-8B-1.58-100B-tokens', '--outtype', 'f32']' died with <Signals.SIGKILL: 9>., check details in logs/convert_to_f32_gguf.log
```

続いて、Desktopマシンでも20Gibのswap fileを作成してから`HF1BitLLM/Llama3-8B-1.58-100B-tokens`の変換処理を行なうと、
無事成功し、このモデルもちゃんと動作した。
