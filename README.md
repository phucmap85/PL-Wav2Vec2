# LingWav2Vec2 + pitch encoder for Vietnamese MDD Challenge

The model uses a Wav2Vec2 acoustic backbone, extracts NCCF-based pitch features, encodes them with `PitchEncoder`, and combines the acoustic sequence with canonical phoneme information through cross-attention before the CTC prediction head.

On the private test set, it achieved a score of `0.6990` and ranked in the top 3.

## Installation

Create a Python environment and install the dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Run commands from the repository root with `python cli.py`:

```bash
python cli.py --help
```

## Configuration

All main paths and hyperparameters are stored in:

```text
config.json
```

The default configuration uses:

- `metadata/train_phones.csv` for training
- `metadata/eval_phones.csv` for evaluation
- `vocab.json` as the phone vocabulary
- `nguyenvulebinh/wav2vec2-base-vietnamese-250h` as the base acoustic model
- `outputs/wav2vec2-250h-pitch-freeze-w2v` as the output directory

You can override individual config values from the command line with `--set key=value`.

## Build the Vocabulary

```bash
python cli.py make-vocab --input metadata/lexicon_vmd.txt --output vocab.json
```

## Train

Run training with the default configuration:

```bash
python cli.py train --config config.json
```

Override a config value for a run:

```bash
python cli.py train --config config.json --set num_train_epochs=5
```

Model checkpoints, the final model, the feature extractor, and a copy of the vocabulary are saved under the `output_dir` configured in `config.json`.

## Predict

Run prediction from a trained checkpoint:

```bash
python cli.py predict ^
  --config config.json ^
  --checkpoint outputs/wav2vec2-250h-pitch-freeze-w2v ^
  --data metadata/test_phones.csv ^
  --output results.csv ^
  --max-samples 20
```
