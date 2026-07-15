"""Minimal AR-style label decoder for the multilingual OCR recogniser.

Reproduces ``openrec.postprocess.ar_postprocess.ARLabelDecode`` /
``ScriptMoELabelDecode`` so that decoded text matches the training pipeline.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import torch


class ARLabelDecoder:
    """Decode argmax predictions into strings using a multilingual char dict."""

    BOS = '<s>'
    EOS = '</s>'
    PAD = '<pad>'

    def __init__(self, character_dict_path: str, use_space_char: bool = True):
        chars: List[str] = []
        with open(character_dict_path, 'rb') as fin:
            for line in fin.readlines():
                chars.append(line.decode('utf-8').strip('\n').strip('\r\n'))
        if use_space_char:
            chars.append(' ')

        # Mirror ARLabelDecode.add_special_char: [EOS] + dict + [BOS, PAD]
        self.character: List[str] = [self.EOS] + chars + [self.BOS, self.PAD]
        self.dict = {c: i for i, c in enumerate(self.character)}
        # The original code reverses the output for arabic dicts — we keep
        # it off here because dict is a mixed multilingual dict.
        self.reverse = False

    @property
    def num_classes(self) -> int:
        return len(self.character)

    def __call__(self, preds) -> List[Tuple[str, float]]:
        if isinstance(preds, dict):
            preds = preds['logit']
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        return self._decode(preds.argmax(axis=2), preds.max(axis=2))

    # ------------------------------------------------------------------
    def _decode(self, text_index: np.ndarray,
                text_prob: np.ndarray) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for b in range(len(text_index)):
            chars = []
            confs = []
            for j in range(len(text_index[b])):
                idx = int(text_index[b][j])
                if idx >= len(self.character):
                    continue
                c = self.character[idx]
                if c == self.EOS:
                    break
                if c == self.BOS or c == self.PAD:
                    continue
                chars.append(c)
                confs.append(float(text_prob[b][j]))

            text = ''.join(chars)
            if self.reverse:
                text = self._pred_reverse(text)
            score = float(np.mean(confs)) if confs else 0.0
            results.append((text, score))
        return results

    @staticmethod
    def _pred_reverse(pred: str) -> str:
        out, cur = [], ''
        for ch in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', ch)):
                if cur:
                    out.append(cur)
                out.append(ch)
                cur = ''
            else:
                cur += ch
        if cur:
            out.append(cur)
        return ''.join(out[::-1])
