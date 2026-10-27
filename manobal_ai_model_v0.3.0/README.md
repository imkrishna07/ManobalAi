# Manobal-AI model package v0.3.0

This package contains the synthetic LightGBM model trained for Manobal-AI's existing backend contract: feature version 1.1.0, 44 backend-engineered features, and Low/Moderate/High classes.

## Contents

- `artifacts/`: model, baseline, preprocessing, metadata, and model card.
- `backend/lib/`: feature/preprocessing modules required by the application and serialized preprocessing object.
- `backend/train_synthetic_model.py`: reproducible synthetic generator and training/export script.
- `requirements-model.txt`: exact Python package versions used by the training container.

To install in the Manobal-AI checkout, copy the four files from `artifacts/` into the repository's root `artifacts/` directory (and mirror them to `backend/artifacts/` if your workflow uses that duplicate). Keep the backend's `lib.feature_engineering` and `lib.preprocessing` import paths available. The runtime checks the exact feature names and order before serving.

To reproduce from the repository root, run:

```bash
docker compose run --build --rm --no-deps backend python train_synthetic_model.py \
  --output-dir /app/artifacts --mirror-dir /app/model-artifacts-mirror
```

## Limitations

This is a hackathon/demo model trained and evaluated on generated synthetic labels only. Its reported metrics are not real-world performance. It is not production-ready and must not be used for operational personnel decisions. Read `artifacts/MODEL_CARD.md` before use.
