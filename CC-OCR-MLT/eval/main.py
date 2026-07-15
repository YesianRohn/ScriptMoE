"""Vendored entry point for CC-OCR evaluation.
Usage:
    python cc_ocr_eval/main.py <index_path> <exp_dir_path>
"""

import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluator import evaluator_map_info, summary  # noqa: E402


def evaluate_and_summary(index_path, exp_dir_path):
    with open(index_path, "r") as f:
        data_list = json.load(f)

    all_evaluation_info = {}
    res_path = os.path.join(exp_dir_path, "status.json")
    keeper_base = os.path.abspath(os.path.join(os.path.dirname(index_path), ".."))
    for data_info in data_list:
        data_name = data_info["dataset"]
        group_name = data_info["group"]
        if not data_info.get("release", True):
            continue

        data_base_dir = os.path.join(keeper_base, data_info["base_dir"])
        kie_gt_file_path = os.path.join(data_base_dir, "label.json")
        pdt_res_dir_path = os.path.join(exp_dir_path, data_name)
        if not os.path.exists(pdt_res_dir_path):
            print(f"--> skip {data_name}: result dir not found {pdt_res_dir_path}")
            continue
        if not os.path.exists(kie_gt_file_path):
            print(f"--> skip {data_name}: gt label.json not found {kie_gt_file_path}")
            continue

        with open(kie_gt_file_path, "r", encoding="utf-8") as f:
            gt_info = json.load(f)

        eval_func = evaluator_map_info.get(group_name)
        if eval_func is None:
            raise ValueError(f"evaluator not defined for: {group_name}")

        meta_info, eval_info = eval_func(pdt_res_dir_path, gt_info, **data_info)
        formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        all_evaluation_info[data_name] = {
            "config": data_info, "meta": meta_info,
            "evaluation": eval_info, "time": formatted_time,
        }

    print(f"--> exp evaluation results save at: {os.path.abspath(res_path)}")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(all_evaluation_info, f, ensure_ascii=False, indent=4)

    exp_dir_base = os.path.dirname(os.path.abspath(exp_dir_path))
    summary_path = summary(index_path, exp_dir_base)
    return summary_path


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} index_path exp_dir_path")
        sys.exit(1)
    summary_path = evaluate_and_summary(sys.argv[1], sys.argv[2])
    print(f"--> info: summary saved at : {summary_path}")
