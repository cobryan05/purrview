# Training Container Tests

These tests **must run inside the `purrview-training-dev` container** where
PyTorch, torchvision, timm, albumentations, and onnxruntime-gpu are installed.

```bash
# Inside the training container:
pytest tests_training/ -v
```

Do NOT run these from the dev/inference container — they will fail on import.

## E2E tests (`test_e2e_*.py`)

E2E tests call the training server's live HTTP API.  They require:
- The training container to be running on port 8005
- The db_server to be running (for label queries)
- At least one label with present/absent images in the DB

```bash
# Inside the training container (or from host with port 8005 exposed):
pytest tests_training/test_e2e_jobs.py -v
```
