# Fine-tuning Checklist

1. **Data cleaning** – normalize the encoding (UTF-8), remove sensitive data, and comment out irrelevant fragments.
2. **Split** – keep 90% of the data for `corpus_train.txt` and 10% for `corpus_eval.txt`.
3. **Block size** – maintain 1,024 tokens for GPT-2; adjust for models with larger context (e.g., 4,096 for Qwen2.5-Coder 7B).
4. **Batch & gradient accumulation** – set a small `batch-size` (1–2) and high `gradient-accumulation-steps` to simulate large batches without exhausting memory.
5. **Monitoring** – track `eval_loss` and `perplexity` at each `eval_steps`; stop training if performance stagnates or increases.
