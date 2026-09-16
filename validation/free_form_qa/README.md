# Blinded free-form QA evaluation

100 questions, 200 answers, and 400 answer ratings from two reviewers. Anonymous CSV ratings and the A/B answer key are in `human_review/`.
The Vanilla control disables components while retaining the mixed index. Citation validity was not separately rated by reviewers.

```bash
python3 validation/free_form_qa/scripts/analyze_human_review_workbooks.py
```
