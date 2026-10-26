# Jev baseline

Reproduces the **Jev** row of the main results table (Table 3 of the paper). [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe's System One model: it returns a probability for a yes/no question instead of generating text. We run it zero-shot (version `jev-1.13.0`) on the same text input as PaperScorer-Text and ask whether the paper will be accepted.

Two scripts, standard library only:

- `run_jev.py` calls the Jev API and writes one JSONL row per paper: `{"idx", "paper_id", "gold", "noul", "model", "input_tokens"}`, where `noul` is P(accept).
- `score.py` fits the accept cutoff on the balanced validation set (maximizing balanced accuracy), applies it to the balanced test set, and prints Acc, AUC, R_A/R_R, p_A and rho_R.

## 1. Get the text inputs

The eval sets are rebuilt from the Hugging Face dataset [`skonan/PaperScorer-Text`](https://huggingface.co/datasets/skonan/PaperScorer-Text) with the same reconstruction script the training code uses:

```bash
cd paperscorer-training-and-inference
uv sync
uv run python scripts/reconstruction.py --hf_text_repo skonan/PaperScorer-Text --dataset_keys \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_validation \
    iclr_2020_2023_2025_2026_85_5_10_balanced_original_text_labelfix_v7_filtered_y25up_test \
    arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_validation \
    arxiv_50_50_21k_text_wmetadata_filtered24480_y24up_test
```

This writes `data/<key>/data.json` (sharegpt rows). `run_jev.py` strips PaperScorer's instruction preamble, sends the paper as Jev's `state`, and asks one Noul question that carries the same information (venue and typical acceptance rate).

## 2. Query Jev

```bash
cd tools/jev-baseline
export TYPESAFE_API_KEY=...        # never commit the key
python run_jev.py --data_root ../../paperscorer-training-and-inference/data --sets all
```

Rows land in `outputs/<set>.jsonl`; rerunning skips papers already scored. Use `--limit 50` for a cheap pilot. The four balanced sets are about 4.6K papers.

## 3. Score

```bash
python score.py --jev_dir outputs --data_root ../../paperscorer-training-and-inference/data
```

`--data_root` is only needed for rho_R (Spearman correlation with the ICLR reviewer-rating percentile). Expected output on the paper's runs:

```
dataset  cutoff   Acc   AUC  R_A/R_R   p_A  rho_R n_test failed
ICLR       0.32  63.4 0.682   81/45   67.9   +.37   1667      0
arXiv      0.29  60.6 0.659   78/44   66.8      -   1415      0
```

Acc is balanced accuracy, and all numbers are on the balanced test sets. Jev's raw probabilities sit mostly below 0.5, which is why the cutoff is fit on validation; at a fixed 0.5 cutoff its Acc is near chance (50.1% ICLR, 50.6% arXiv).
