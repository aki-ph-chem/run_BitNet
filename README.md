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

自分の環境では、HFからGGUFへの変換でコケてる。

```txt
$ poetry run python setup_env.py --hf-repo HF1BitLLM/Llama3-8B-1.58-100B-tokens -q i2_s
INFO:root:Compiling the code using CMake.
INFO:root:Downloading model HF1BitLLM/Llama3-8B-1.58-100B-tokens from HuggingFace to models/Llama3-8B-1.58-100B-tokens...
INFO:root:Converting HF model to GGUF format...
ERROR:root:Error occurred while running command: Command '['/home/aki/run_BitNet/BitNet/.venv/bin/python', 'utils/convert-hf-to-gguf-bitnet.py', 'models/Llama3-8B-1.58-100B-tokens', '--outtype', 'f32']' died with <Signals.SIGKILL: 9>., check details in logs/convert_to_f32_gguf.log
```

ログを読んでも原因が分らない！

同じような症状は[issue](https://github.com/microsoft/BitNet/issues/27)にもあった。

デフォルトの`HF1BitLLM/Llama3-8B-1.58-100B-tokens`は結局モデルがかなり大規模らしく、モデル変換でメモリが足りなくなって、死んでしまうらしい。
そこでより、小さなモデルである3Bのモデルを試すことにした。

```bash
poetry run python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-3B -q i2_s # 結構時間がかかる
```

こっちは、うまくビルドできた。

## ついに動かす!

```bash
python run_inference.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf -p "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:" -n 6 -temp 0
```

一応動いた、けどanswerが空欄で返答がない。
```bash
warning: not compiled with GPU offload support, --gpu-layers option will be ignored
warning: see main README.md for information on enabling GPU BLAS support
build: 3947 (406a5036) with clang version 18.1.8 for x86_64-pc-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_loader: loaded meta data with 26 key-value pairs and 288 tensors from models/bitnet_b1_58-3B/ggml-model-i2_s.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet
llama_model_loader: - kv   1:                               general.name str              = bitnet_b1_58-3B
llama_model_loader: - kv   2:                         bitnet.block_count u32              = 26
llama_model_loader: - kv   3:                      bitnet.context_length u32              = 2048
llama_model_loader: - kv   4:                    bitnet.embedding_length u32              = 3200
llama_model_loader: - kv   5:                 bitnet.feed_forward_length u32              = 8640
llama_model_loader: - kv   6:                bitnet.attention.head_count u32              = 32
llama_model_loader: - kv   7:             bitnet.attention.head_count_kv u32              = 32
llama_model_loader: - kv   8:                      bitnet.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9:    bitnet.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 40
llama_model_loader: - kv  11:                          bitnet.vocab_size u32              = 32002
llama_model_loader: - kv  12:                   bitnet.rope.scaling.type str              = linear
llama_model_loader: - kv  13:                 bitnet.rope.scaling.factor f32              = 1.000000
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  15:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  16:                      tokenizer.ggml.tokens arr[str,32002]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  17:                      tokenizer.ggml.scores arr[f32,32002]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  18:                  tokenizer.ggml.token_type arr[i32,32002]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  19:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  21:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  22:            tokenizer.ggml.padding_token_id u32              = 32000
llama_model_loader: - kv  23:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  24:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  25:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:  105 tensors
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type i2_s:  182 tensors
llm_load_vocab: control token:      2 '</s>' is not marked as EOG
llm_load_vocab: control token:      1 '<s>' is not marked as EOG
llm_load_vocab: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
llm_load_vocab: special tokens cache size = 5
llm_load_vocab: token to piece cache size = 0.1684 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = bitnet
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32002
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 3200
llm_load_print_meta: n_layer          = 26
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_rot            = 100
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 100
llm_load_print_meta: n_embd_head_v    = 100
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 3200
llm_load_print_meta: n_embd_v_gqa     = 3200
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 8640
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 2048
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 3B
llm_load_print_meta: model ftype      = I2_S - 2 bpw ternary
llm_load_print_meta: model params     = 3.32 B
llm_load_print_meta: model size       = 873.66 MiB (2.20 BPW)
llm_load_print_meta: general.name     = bitnet_b1_58-3B
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 32000 '</line>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_print_meta: EOG token        = 2 '</s>'
llm_load_print_meta: max token length = 48
llm_load_tensors: ggml ctx size =    0.13 MiB
llm_load_tensors:        CPU buffer size =   873.66 MiB
..........................................................................................
llama_new_context_with_model: n_batch is less than GGML_KQ_MASK_PAD - increasing to 32
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 32
llama_new_context_with_model: n_ubatch   = 32
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   650.00 MiB
llama_new_context_with_model: KV self size  =  650.00 MiB, K (f16):  325.00 MiB, V (f16):  325.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
llama_new_context_with_model:        CPU compute buffer size =     9.81 MiB
llama_new_context_with_model: graph nodes  = 942
llama_new_context_with_model: graph splits = 1
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 2

system_info: n_threads = 2 (n_threads_batch = 2) / 24 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |

sampler seed: 4294967295
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> greedy
generate: n_ctx = 2048, n_batch = 1, n_predict = 6, n_keep = 1

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

- `1bitLLM/bitnet_b1_58-large`

こっちのモデルもビルドできた！
```bash
$ poetry run python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large -q i2_s
INFO:root:Compiling the code using CMake.
INFO:root:Downloading model 1bitLLM/bitnet_b1_58-large from HuggingFace to models/bitnet_b1_58-large...
INFO:root:Converting HF model to GGUF format...
INFO:root:GGUF model saved at models/bitnet_b1_58-large/ggml-model-i2_s.gguf
```

こっちも動かす
```txt
warning: not compiled with GPU offload support, --gpu-layers option will be ignored
warning: see main README.md for information on enabling GPU BLAS support
build: 3947 (406a5036) with clang version 18.1.8 for x86_64-pc-linux-gnu
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_loader: loaded meta data with 26 key-value pairs and 266 tensors from models/bitnet_b1_58-large/ggml-model-i2_s.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = bitnet
llama_model_loader: - kv   1:                               general.name str              = bitnet_b1_58-large
llama_model_loader: - kv   2:                         bitnet.block_count u32              = 24
llama_model_loader: - kv   3:                      bitnet.context_length u32              = 2048
llama_model_loader: - kv   4:                    bitnet.embedding_length u32              = 1536
llama_model_loader: - kv   5:                 bitnet.feed_forward_length u32              = 4096
llama_model_loader: - kv   6:                bitnet.attention.head_count u32              = 16
llama_model_loader: - kv   7:             bitnet.attention.head_count_kv u32              = 16
llama_model_loader: - kv   8:                      bitnet.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv   9:    bitnet.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 40
llama_model_loader: - kv  11:                          bitnet.vocab_size u32              = 32002
llama_model_loader: - kv  12:                   bitnet.rope.scaling.type str              = linear
llama_model_loader: - kv  13:                 bitnet.rope.scaling.factor f32              = 1.000000
llama_model_loader: - kv  14:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  15:                         tokenizer.ggml.pre str              = default
llama_model_loader: - kv  16:                      tokenizer.ggml.tokens arr[str,32002]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  17:                      tokenizer.ggml.scores arr[f32,32002]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  18:                  tokenizer.ggml.token_type arr[i32,32002]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  19:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  20:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  21:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  22:            tokenizer.ggml.padding_token_id u32              = 32000
llama_model_loader: - kv  23:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  24:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  25:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   97 tensors
llama_model_loader: - type  f16:    1 tensors
llama_model_loader: - type i2_s:  168 tensors
llm_load_vocab: control token:      2 '</s>' is not marked as EOG
llm_load_vocab: control token:      1 '<s>' is not marked as EOG
llm_load_vocab: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
llm_load_vocab: special tokens cache size = 5
llm_load_vocab: token to piece cache size = 0.1684 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = bitnet
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32002
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 1536
llm_load_print_meta: n_layer          = 24
llm_load_print_meta: n_head           = 16
llm_load_print_meta: n_head_kv        = 16
llm_load_print_meta: n_rot            = 96
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 96
llm_load_print_meta: n_embd_head_v    = 96
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 1536
llm_load_print_meta: n_embd_v_gqa     = 1536
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 4096
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 2
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 2048
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = 700M
llm_load_print_meta: model ftype      = I2_S - 2 bpw ternary
llm_load_print_meta: model params     = 728.84 M
llm_load_print_meta: model size       = 256.56 MiB (2.95 BPW)
llm_load_print_meta: general.name     = bitnet_b1_58-large
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: PAD token        = 32000 '</line>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_print_meta: EOG token        = 2 '</s>'
llm_load_print_meta: max token length = 48
llm_load_tensors: ggml ctx size =    0.12 MiB
llm_load_tensors:        CPU buffer size =   256.56 MiB
.................................................................
llama_new_context_with_model: n_batch is less than GGML_KQ_MASK_PAD - increasing to 32
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: n_batch    = 32
llama_new_context_with_model: n_ubatch   = 32
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =   288.00 MiB
llama_new_context_with_model: KV self size  =  288.00 MiB, K (f16):  144.00 MiB, V (f16):  144.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.12 MiB
llama_new_context_with_model:        CPU compute buffer size =     5.00 MiB
llama_new_context_with_model: graph nodes  = 870
llama_new_context_with_model: graph splits = 1
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 2

system_info: n_threads = 2 (n_threads_batch = 2) / 24 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | RISCV_VECT = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 |

sampler seed: 4294967295
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.000
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> greedy
generate: n_ctx = 2048, n_batch = 1, n_predict = 6, n_keep = 1

 Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?
Answer: Mary went to the kitchen.

llama_perf_sampler_print:    sampling time =       0.08 ms /    60 runs   (    0.00 ms per token, 731707.32 tokens per second)
llama_perf_context_print:        load time =     118.12 ms
llama_perf_context_print: prompt eval time =    1205.08 ms /    54 tokens (   22.32 ms per token,    44.81 tokens per second)
llama_perf_context_print:        eval time =     116.27 ms /     5 runs   (   23.25 ms per token,    43.00 tokens per second)
llama_perf_context_print:       total time =    1321.96 ms /    59 tokens
```

こっちだと、答が出力されてる！
