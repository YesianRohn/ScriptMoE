import os
import json
import unicodedata
import re
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
import regex
from bidi.algorithm import get_display

EPS = 1e-12


def _strip_spaces(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("\r", " ").replace("\n", " ").replace(" ", "").strip()


def _normalize_filter_symbols_like_metric(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = regex.sub(r"\s+", "", s)
    s = regex.sub(r"[\p{P}\p{S}]+", "", s)
    return s


class MetricAgg:
    def __init__(self):
        self.correct_num = 0
        self.correct_num_real = 0
        self.correct_num_lower = 0
        self.correct_num_ignore_space = 0
        self.correct_num_ignore_space_lower = 0
        self.correct_num_ignore_space_symbol = 0
        self.all_num = 0
        self.norm_edit_dis_sum = 0.0

    def add_one(self, pred: str, gt: str,
                final_ignore_space=False,
                final_is_filter=False,
                final_is_lower=False):

        if pred is None:
            pred = ""
        if gt is None:
            gt = ""
        pred = str(pred)
        gt = str(gt)

        if pred == gt:
            self.correct_num_real += 1
        if pred.lower() == gt.lower():
            self.correct_num_lower += 1

        pred_ns = _strip_spaces(pred)
        gt_ns = _strip_spaces(gt)
        if pred_ns == gt_ns:
            self.correct_num_ignore_space += 1
        if pred_ns.lower() == gt_ns.lower():
            self.correct_num_ignore_space_lower += 1

        pred_sym = _normalize_filter_symbols_like_metric(pred)
        gt_sym = _normalize_filter_symbols_like_metric(gt)
        if pred_sym == gt_sym:
            self.correct_num_ignore_space_symbol += 1

        # 归一化编辑距离（保持你原代码：用“未 final 处理”的 pred/gt 计算）
        dis = Levenshtein.normalized_distance(pred, gt)
        self.norm_edit_dis_sum += dis

        pred_final = pred
        gt_final = gt
        if final_ignore_space:
            pred_final = pred_final.replace(" ", "")
            gt_final = gt_final.replace(" ", "")
        if final_is_filter:
            pred_final = _normalize_filter_symbols_like_metric(pred_final)
            gt_final = _normalize_filter_symbols_like_metric(gt_final)
        if final_is_lower:
            pred_final = pred_final.lower()
            gt_final = gt_final.lower()

        if pred_final == gt_final:
            self.correct_num += 1

        self.all_num += 1

    def summary(self):
        n = self.all_num
        return {
            "num_samples": n,
            "acc": self.correct_num / (n + EPS),
            "acc_real": self.correct_num_real / (n + EPS),
            "acc_lower": self.correct_num_lower / (n + EPS),
            "acc_ignore_space": self.correct_num_ignore_space / (n + EPS),
            "acc_ignore_space_lower": self.correct_num_ignore_space_lower / (n + EPS),
            "acc_ignore_space_symbol": self.correct_num_ignore_space_symbol / (n + EPS),
            "norm_edit_dis": 1.0 - (self.norm_edit_dis_sum / (n + EPS)),
        }


def eval_jsonl_files(
    jsonl_paths,
    final_ignore_space: bool = False,
    final_is_filter: bool = False,
    final_is_lower: bool = False,
    verbose_errors_each: int = 0,
    group_key: str = "dataset",  # 也可改成 "lmdb_path"
):
    aggs = defaultdict(MetricAgg)
    skipped_cnt = defaultdict(int)
    shown_err = defaultdict(int)

    for jp in jsonl_paths:
        with open(jp, "r", encoding="utf-8") as fr:
            for line in fr:
                if not line.strip():
                    continue
                row = json.loads(line)
                ds = row.get(group_key, os.path.basename(jp))
                if row.get("skipped", False):
                    skipped_cnt[ds] += 1
                    continue

                gt = row.get("gt", "")
                pred = row.get("pred", "")

                aggs[ds].add_one(
                    pred, gt,
                    final_ignore_space=final_ignore_space,
                    final_is_filter=final_is_filter,
                    final_is_lower=final_is_lower,
                )

                if verbose_errors_each > 0 and shown_err[ds] < verbose_errors_each:
                    # 以最终口径判断是否错
                    pred_f, gt_f = pred, gt
                    if final_ignore_space:
                        pred_f = pred_f.replace(" ", "")
                        gt_f = gt_f.replace(" ", "")
                    if final_is_filter:
                        pred_f = _normalize_filter_symbols_like_metric(pred_f)
                        gt_f = _normalize_filter_symbols_like_metric(gt_f)
                    if final_is_lower:
                        pred_f = pred_f.lower()
                        gt_f = gt_f.lower()

                    if pred_f != gt_f:
                        print(f"[ERR][{ds}] gt='{gt}' pred='{pred}' (idx={row.get('idx')})")
                        shown_err[ds] += 1

    # 输出每个数据集
    results = {}
    for ds, agg in aggs.items():
        metrics = agg.summary()
        results[ds] = {"metrics": metrics, "skipped": skipped_cnt[ds]}

    return results


if __name__ == "__main__":
    # 读取某目录下所有 jsonl
    dump_dir = "./qwen3_5_9b_results"
    jsonl_files = [
        os.path.join(dump_dir, f) for f in os.listdir(dump_dir) if f.endswith(".jsonl")
    ]
    jsonl_files.sort()

    results = eval_jsonl_files(
        jsonl_files,
        final_ignore_space=True,
        final_is_filter=True,
        final_is_lower=True,
        verbose_errors_each=5,
        group_key="dataset",
    )

    for ds in sorted(results.keys()):
        m = results[ds]["metrics"]
        skipped = results[ds]["skipped"]
        print(f"[DATASET] {ds}")
        print(
            f"  acc={m['acc']:.4f}  "
            f"acc_real={m['acc_real']:.4f}  "
            f"acc_lower={m['acc_lower']:.4f}  "
            f"acc_ignore_space={m['acc_ignore_space']:.4f}  "
            f"acc_ignore_space_lower={m['acc_ignore_space_lower']:.4f}  "
            f"acc_ignore_space_symbol={m['acc_ignore_space_symbol']:.4f}  "
            f"norm_edit_dis={m['norm_edit_dis']:.4f}  "
            f"num_samples={m['num_samples']}  skipped={skipped}"
        )