# Insurance Fraud Detection

MLE Portfolio Project - Project structure initialized

## Running modules (fixes relative import errors)

If you see:

`ImportError: attempted relative import with no known parent package`

run entrypoints as modules from the repo root:

```bash
python -m src.api.app
```

Or run the API via Uvicorn:

```bash
uvicorn src.api.app:app --reload --port 8000
```

