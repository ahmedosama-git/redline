# Redline

Small grammar checkers built by fully fine-tuning [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on a single T4 GPU. There are two models:

| Model | What it does | Headline result |
|---|---|---|
| [**redline-qwen3-0.6b**](https://huggingface.co/ahmedoussama/redline-qwen3-0.6b) | General English: returns the corrected sentence | **59.02 GLEU** on JFLEG test, vs 55.60 for `grammarly/coedit-large` (770M) and 61.12 for human annotators |
| [**redline-bank-qwen3-0.6b**](https://huggingface.co/ahmedoussama/redline-bank-qwen3-0.6b) | Banking and financial text: returns a JSON list of edits and leaves financial and legal terminology alone | **F0.5 0.742** on banking text, vs 0.379 for coedit-large; valid JSON 100% of the time |

All training and evaluation runs on free Kaggle notebooks, and every number in this README comes from the public notebook runs linked below.

## Repository layout

```
general/                     general-purpose grammar corrector
  grammar/                   data → full fine-tune → JFLEG + held-out evaluation
  decode/                    decoding sweep (greedy vs beam search)
banking/                     banking-domain grammar checker
  train/                     pandas-vs-Spark benchmark → Spark data pipeline → fine-tune → evaluation
  reeval/                    re-scores the saved checkpoint with a corrected evaluation
  jfleg/                     scores the banking model on JFLEG (human learner errors)
```

Each folder holds one Kaggle notebook and its `kernel-metadata.json`, so `kaggle kernels push -p <folder>` runs it as-is.

| Notebook | Kaggle | Depends on |
|---|---|---|
| `general/grammar` | [qwen3-grammar-checker](https://www.kaggle.com/code/ososovic/qwen3-grammar-checker) | – |
| `general/decode` | [qwen3-gec-decoding](https://www.kaggle.com/code/ososovic/qwen3-gec-decoding) | grammar's output model |
| `banking/train` | [qwen3-bank-grammar-pandas-vs-spark](https://www.kaggle.com/code/ososovic/qwen3-bank-grammar-pandas-vs-spark) | CFPB complaints dataset |
| `banking/reeval` | [qwen3-bank-grammar-reeval](https://www.kaggle.com/code/ososovic/qwen3-bank-grammar-reeval) | train's output model |
| `banking/jfleg` | [qwen3-bank-grammar-jfleg](https://www.kaggle.com/code/ososovic/qwen3-bank-grammar-jfleg) | train's output model |

## General model

**Task.** Sentence in, corrected sentence out. An earlier version of this project trained the model to emit an edit list instead, the format the banking model still uses. That forces the model to reproduce exact source spans, where one wrong boundary breaks the whole correction. Emitting the corrected text is an easier task for the model. When a UI needs highlighted edits, they are derived from the input and output with a word-level diff, so every edit span is a real substring of the input.

**Data.** 149,843 correction pairs: W&I+LOCNESS (23,471, human-annotated), CoEdIT (18,000, human-annotated) and C4-200M (108,372, synthetic). On top of these come 17,981 sentences that are already correct, so the model learns that "leave it alone" is often the right answer. Pairs are kept only if the diff round-trips exactly. Train and dev are split on a hash of the content, so adding data later can never move a training example into dev.

**Results.**

| | coedit-large | Redline |
|---|---|---|
| JFLEG test GLEU (greedy) | 55.60 | **59.02** |
| JFLEG test GLEU (4-beam, chosen on validation) | – | **59.26** |
| Held-out F0.5 | 0.277 | **0.445** |
| No-op rate ↓ (rewrites already-correct text) | 0.628 | **0.129** |

GLEU is implemented from the paper rather than taken from `nltk`, whose Google-BLEU ignores the source sentence. Before scoring, it is checked against the published anchors (source ≈ 40.5, human ≈ 62), and the notebook aborts if they disagree.

## Banking model

**Task.** Grammar, spelling and punctuation checking for customer letters, complaint replies, notices and agreement clauses. Instruments, defined terms, figures and legal phrasing must not be "corrected". The output is a JSON edit list (`[{"o": "were", "c": "was", "n": 1}]`, or `[]` when the text is correct) that a UI can render as suggestions.

**Data pipeline: pandas vs Spark.** The notebook first benchmarks both engines on the same logical pipeline over the same Parquet file, so the choice between them is measured rather than assumed:

| Engine | Time | Output rows |
|---|---|---|
| pandas | 1.8 s | 18,730 |
| Spark (local mode) | 4.9 s | 18,730 |

At this size, Spark's JVM, scheduling and serialisation overhead dominates. The final pipeline is therefore hybrid. Spark handles the large tabular stages: about 4.06M sentences from CFPB complaints, SEC EDGAR filings and seed text, plus the cleaning, union, deduplication, split and Parquet output. Python handles the custom diff and error-injection logic on bounded samples. Errors are injected by nine corruption functions, which skip a curated list of protected banking terms.

**Results** (400 banking sentences with errors and 400 already-correct ones):

| | coedit-large | Redline Bank |
|---|---|---|
| F0.5 | 0.379 | **0.742** |
| Exact match | 0.117 | **0.455** |
| No-op rate ↓ | 0.725 | **0.147** |
| Valid JSON | – | 1.000 |

An independently built banking set in `reeval` gives F0.5 0.775 vs 0.409.

**Limitations.** The banking errors above are synthetic. On JFLEG's human learner errors, the banking model scores 51.26 GLEU against coedit-large's 55.63. Specialising costs general-domain quality, which is why the general model exists.

## Engineering notes

- **Evaluation fixes.** `reeval` exists because the first banking evaluation was misleading. It blended a ~95% general-domain set with banking text. It also gave the baseline free canonicalisation that the fine-tuned model did not get, and it prompted with a different separator than the one used in training. With those fixed, the result went from "the fine-tune loses" to "the fine-tune wins clearly in its domain".
- **Pinned dependencies.** LLaMA-Factory is pinned to commit `b4e4d3f` and transformers to `5.15.1`. Unpinned, the two drifted apart and broke training twice: transformers 5.x removed `warmup_ratio`, and transformers ≥ 5.16 passes `every_n_layers` to a gradient-checkpointing hook that LLaMA-Factory overrides.
- **Fitting a T4.** A full fine-tune with 32-bit Adam needs about 13.5 GiB and does not fit a 14.6 GiB T4, so training uses Adafactor with fp16 and gradient checkpointing. Two-GPU DDP deadlocked on this Kaggle image, so training runs on one GPU: about 5 h 15 min per model.
- **Smoke mode.** Each training notebook has a `SMOKE` flag that runs the whole pipeline in minutes, so a broken cell shows up before a multi-hour run.

## Reproduce

1. Create a Kaggle account with phone verification (needed for GPU and internet access).
2. Change the username in `id` and `kernel_sources` in each `kernel-metadata.json` to your own.
3. Push the training notebooks first, then the notebooks that depend on them:

```bash
kaggle kernels push -p general/grammar
kaggle kernels push -p banking/train
# after they finish:
kaggle kernels push -p general/decode
kaggle kernels push -p banking/reeval
kaggle kernels push -p banking/jfleg
```

Set `SMOKE = True` in the first cells to test the pipeline quickly.

## License

- **Code:** [PolyForm Noncommercial 1.0.0](LICENSE)
- **Models:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

You are free to read, learn from and use this work for research, teaching and personal projects. **Production or commercial use requires written permission.** To ask, contact me through [my GitHub profile](https://github.com/ahmedosama-git).

The models build on [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache-2.0). Training data includes CoEdIT (Apache-2.0), W&I+LOCNESS (research use) and C4-200M. The banking model also uses public CFPB complaint data and SEC EDGAR filings.
