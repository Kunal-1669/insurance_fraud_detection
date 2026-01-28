from setuptools import setup, find_packages

setup(
    name="insurance_fraud_detection",
    author="Kunal Inglunkar",
    author_email="kunalinglunkar@gmail.com",
    description="Insurance Fraud Detection System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/insurance-fraud-detection",
    # This repo uses `src/` as an actual top-level Python package (i.e. `import src...`).
    # That allows relative imports inside `src/*` modules (e.g. `from ..utils...`) to work
    # when running entrypoints like `python -m src.api.app`.
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "mlflow>=2.9.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
    },
)
