# tediet

拡散モデルパイプラインのテキストエンコーダをダイエットさせるライブラリです。

[English README](README_EN.md)

最近の画像・動画生成モデルは巨大なテキストエンコーダ — T5-XXL(約9GB)、
Gemma、Qwen3-VL(約16GB)— を抱えていますが、テキストエンコーダは
**1ジョブに1回しか走りません**。拡散トランスフォーマが全ステップ回る間、
エンコーダ全体をGPUに常駐させるのは、16〜24GB級GPUの大半を無駄にします。
`tediet` は、その常駐を取り戻す、小さく・組み合わせ可能で・**出力が
ビット一致する** 2つの変換を提供します。汎用CPUオフロードのような
レイテンシの犠牲はありません。

| 手法 | 何をするか | VRAM効果 |
|---|---|---|
| **diet** | トークン埋め込みテーブルと(未使用の)LMヘッドをCPUへ移し、全語彙logitsの計算をスキップする | Qwen3-VLで−2.3 GiB、Gemmaで−1.9 GiB |
| **stream** | デコーダ層をpinnedホストメモリに常駐させ、エンコード実行中だけ、計算の2層先読みで固定GPUリングバッファへコピーする | 層スタックの常駐 12.9 GiB → 約1.3 GiB(Qwen3-VL bf16) |

どちらも**ビット一致**です: 隠れ状態は全常駐モデルと完全に同一になります。
計算は何も変えておらず、変わるのは「2つのルックアップがどこに住むか」
「未使用の射影を実行するか」「変化しない重みをどこに置くか」だけだからです。

## 実測

Qwen-Image 2.1(Qwen3-VL 16GBテキストエンコーダ、bf16)、
RTX PRO 4000 Blackwell 24GB、1024×1024、1枚あたりのエンドツーエンド:

| 構成 | テキストエンコーダ常駐 | e2e時間 |
|---|---|---|
| 全常駐 | 16.3 GiB | 基準 |
| diet + stream(窓2) | **約1.5 GiB** | **+0.6秒** |

同じQwen3-VLに対する diffusers 公式 `apply_group_offloading` との比較
(`benchmarks/bench_text_encoder.py` で再現可能。27トークンのプロンプト、
エンコード単体、5回平均):

| 実装 | 常駐 | エンコード | ピーク |
|---|---|---|---|
| 全常駐 | 16.34 GiB | 0.040秒 | 16.39 GiB |
| **tediet(diet + stream 窓2)** | **2.19 GiB** | **0.886秒** | **2.20 GiB** |
| 公式 leaf_level + `use_stream=True` | 2.28 GiB | 1.032秒 | 3.64 GiB |
| 公式 leaf_level(streamなし) | 2.28 GiB | 6.959秒 | 3.44 GiB |

公式実装との差は条件によって1.2倍〜8倍です。公平を期すと: 公式も
leaf_level+streamまで設定を詰めれば近い水準に来ます。ただしtedietは
それに対してもエンコードが約17%速く、ピークが4割低く(固定リング
バッファで新規確保ゼロ)、素朴に適用した場合の罠(エンコーダ全体への
block_levelはベースモデル全体を1グループとして全載せ、内側サブ
モジュールへのblock_levelは埋め込み経路のdevice mismatchでクラッシュ)
がありません。LTX-2.5のGemma NF4エンコーダでは公式比3〜6倍の差を
実測しています。

この差は3つの設計判断から生まれています: フックをオフロードグループ単位
でなく層単位に直付けすること、層の復元をポインタ差し替えにすること
(重みは変化しないのでGPU→CPUコピーは発生しません)、リングバッファに
よりエンコードが**新規GPUメモリを一切確保しない**こと。最後の性質は、
CUDA Graphのメモリプールと同居してもアロケータが安定するという利点も
もたらします。

## 使い方

```python
from tediet import apply_diet, apply_stream

# Qwen3-VL(Qwen-Image 2.x系パイプライン)
freed = apply_diet(pipe.text_encoder, embed_path="model.language_model.embed_tokens")
pinned = apply_stream(pipe.text_encoder, layers_path="model.language_model.layers",
                      device="cuda:0", window=2)
```

2つの呼び出しはどちらの順でも組み合わせられ、それぞれ冪等です。
`apply_diet` はパイプラインが `outputs.hidden_states` しか読まないことを
前提にします(diffusersのテキストエンコーダ経路はすべてそうです)。
`apply_stream` はリングバッファを層シグネチャごとに持つため、構造の
異なる層が交互に並ぶモデルでも動きます。

### モデル別レシピ

| モデル(パイプライン) | `embed_path` | `layers_path` |
|---|---|---|
| Qwen3-VL(Qwen-Image 2.x) | `model.language_model.embed_tokens` | `model.language_model.layers` |
| Gemma(LTX-2.x) | `model.language_model.embed_tokens` | `model.language_model.layers` |
| T5-XXL(FLUX、SD3) | `encoder.embed_tokens`¹ | `encoder.block` |

¹ T5はエンコーダ専用モデルです: `lm_head_path=None` を渡し、
`apply_lm_head_skip` は使いません。

3レシピとも実機でビット一致を検証済みです(`benchmarks/bench_text_encoder.py`):

| モデル | 常駐(適用前→後) | エンコード(前→後) |
|---|---|---|
| Qwen3-VL bf16(Qwen-Image 2.1) | 16.34 → 2.19 GiB | 0.040 → 0.886秒 |
| Gemma NF4(LTX-2.5) | 7.63 → 0.80 GiB | 0.069 → 0.405秒 |
| T5-XXL bf16(エンコーダ単体) | 8.87 → 1.47 GiB | 0.023 → 0.570秒 |

Gemma NF4では、公式group offloadingはleaf_level・block_levelとも
device mismatchでクラッシュしますが、tedietはそのまま動きます。逆に
素のT5エンコーダ単体では公式のleaf_level+streamも健闘します
(0.633秒、常駐はtedietより低い)。層構造が
交互型のモデル(Gemmaのsliding/globalアテンション層など)にも、リング
バッファを層シグネチャごとに分けることで対応しています。

より多くのモデルのレシピと設計解説は [docs/](docs/) にあります。

## 前提条件と制約

- モデル本体は**常駐**が前提です: `enable_model_cpu_offload` /
  `enable_sequential_cpu_offload` / accelerateフックと併用しないでください。
  これらはコンポーネント全体を `.to()` で往復させるため、CPU配置と衝突します。
- `stream` は層スタックと同量のpinnedホストRAMを使います
  (Qwen3-VL bf16で12.9GB)。
- 対応する層パラメータ: 素のfp32/bf16/fp16テンソルと bitsandbytes 4bit
  (`quant_state` のテンソルも層と一緒に移動します)。TorchAOのtensor
  subclassは未検証です。
- 複数スレッドからの同時エンコードには対応していません。

## ライセンス

MIT
