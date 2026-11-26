# JUDGEBERT-RTI Streamlit UI (checkpoint-matched)

This UI is based on the architecture and checkpoint format in `train_mbert_improved_v1.py`.

## What it does
- Takes an original and simplified legal sentence in English or Hindi.
- Loads a selected trained run locally (does not retrain).
- Uses the run's saved `encoder/` tokenizer/model and `best_checkpoint.pt`.
- Reconstructs the matching CLS or masked-mean pooling regression model.
- Displays a predicted score clipped to 1–10 and the project's operational label.
- Provides a manual review checklist and CSV export.

## Important model-selection note
The training script produced 12 runs (4 configurations × 3 seeds). The app defaults to
`cls_huber_seed123`, which had the lowest recorded validation RMSE among the 12 runs (0.4782). The CLS + Huber configuration also had the lowest mean validation RMSE across seeds. This is a single-seed checkpoint, not an ensemble.

The training script says best epoch is selected by validation RMSE; do not choose a model
based on test metrics. If your final report selected a different run/configuration, change
the folder in the sidebar to that run's folder.

## Required project layout
Place `app.py` in the project root (the same level as `output/`) and ensure:

```
project-root/
  app.py
  output/
    mbert_improved_v1/
      checkpoints/
        cls_mse_seed42/
          best_checkpoint.pt
          encoder/
            config.json
            tokenizer files...
            model weights...
```

You may choose another run folder in the app's sidebar, for example:
- `output/mbert_improved_v1/checkpoints/cls_mse_seed123`
- `output/mbert_improved_v1/checkpoints/cls_mse_seed2026`
- `output/mbert_improved_v1/checkpoints/mean_huber_seed42`

Each selected folder must contain both `best_checkpoint.pt` and `encoder/`.

## Install and run (Windows / VS Code)
Open a terminal at the project root:

```
pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit will print a local address, usually `http://localhost:8501`.

## Notes
- The model is loaded from local files only. The app does not download another model.
- If loading fails, confirm the run folder contains the files shown above and that the
  installed PyTorch/Transformers versions are compatible with your training environment.
- The score is a research prediction, not legal advice or legal certification.
- Thresholds `<6`, `6–<8`, and `>=8` are project-defined labels, not official standards.

## Evaluate how predictive the loaded checkpoint is

Copy `evaluate_checkpoint.py` to the project root. In the script, set `RUN_DIR` to the same run folder that your app is using. Then run:

```
python evaluate_checkpoint.py
```

It evaluates that checkpoint on the held-out `Split=test` rows, reports Pearson/Spearman, MAE/RMSE, prediction spread, and compares against a baseline that predicts the training-set mean. It also reports project-label accuracy/macro-F1 and saves row-level predictions.

Do not use test results to select a model repeatedly. Choose the run using validation RMSE first, then treat the test metrics as final evaluation.
