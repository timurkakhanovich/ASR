# Automatic speech recognition

ML pipeline for training Automatic Speech Recognition based on [Libri Speech dataset](https://paperswithcode.com/dataset/librispeech).

# Prerequisites
1. Ubuntu ^20.04;
2. Python ^3.9 (can be set up with [pyenv](https://github.com/pyenv/pyenv)).


# Installation (for python)
## 1. Create and activate python virtual environment
```bash
python3 -m venv .venv && source .venv/bin/activate
```

## 2. Install all necessary packages with poetry
```bash
pip install -r requirements.txt
```

# Usage
## 1. Data retrieving
Download Libri Speech dataset from this [page](https://www.openslr.org/12), unzip it and place to data directory:
```bash
mkdir Dataset
mv ${DATASET_FOLDERS?} Dataset/
```

## 2. Model training
Run training script with required arguments
```bash
python main.py \
    --checkpoint-path ${CHECKPOINT-PATH?} \
    --checkpoint ${FILE?} \
    --batch-size ${BATCH-SIZE?} \
    --lr ${LR?} \
    --num-epochs ${NUM-EPOCHS?}
```

# Development
## 1. Install pre-commit under your virtual environment
```bash
pip install pre-commit && pre-commit install
```

## 2. Run pre-commit hooks
```bash
pre-commit run --all
```
