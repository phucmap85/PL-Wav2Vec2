from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PAD_TOKEN = "<pad>"


def split_vocab_tokens(text: str) -> List[str]:
    """Split a whitespace-separated phone-token list."""
    return [token for token in text.split() if token]


def split_lexicon_phones(text: str, pad_token: str = PAD_TOKEN) -> List[str]:
    """Extract unique phone tokens from `word phone1 phone2 ...` lexicon lines."""
    phones = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Lexicon line {line_no} must contain a word and at least one phone.")
        phones.update(parts[1:])

    if not phones:
        raise ValueError("No phone tokens found in lexicon.")
    phones.discard(pad_token)
    return [pad_token, *sorted(phones)]


def tokens_to_vocab(tokens: Sequence[str]) -> Dict[str, int]:
    seen = set()
    duplicates = []
    for token in tokens:
        if token in seen:
            duplicates.append(token)
        seen.add(token)
    if duplicates:
        raise ValueError(f"Duplicate token(s) in vocab: {sorted(set(duplicates))}")
    return {token: idx for idx, token in enumerate(tokens)}


def load_vocab(vocab_path: Path | str) -> Tuple[Dict[str, int], List[str]]:
    path = Path(vocab_path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            token_to_id = {token: int(idx) for token, idx in data.items()}
            id_to_token = [None] * len(token_to_id)
            for token, idx in token_to_id.items():
                if idx < 0 or idx >= len(id_to_token):
                    raise ValueError(f"Vocab id out of range for token {token!r}: {idx}")
                if id_to_token[idx] is not None:
                    raise ValueError(f"Duplicate vocab id {idx}")
                id_to_token[idx] = token
            if any(token is None for token in id_to_token):
                raise ValueError("Vocab ids must be contiguous from 0 to len(vocab)-1.")
            return token_to_id, list(id_to_token)
        if isinstance(data, list):
            tokens = [str(token) for token in data]
            return tokens_to_vocab(tokens), tokens
        raise ValueError(f"Unsupported vocab JSON format: {path}")

    tokens = split_vocab_tokens(path.read_text(encoding="utf-8"))
    return tokens_to_vocab(tokens), tokens


def build_vocab_file(input_path: Path | str, output_path: Path | str) -> Dict[str, int]:
    input_file = Path(input_path)
    text = input_file.read_text(encoding="utf-8")
    tokens = split_lexicon_phones(text)
    vocab = tokens_to_vocab(tokens)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return vocab
