"""Vendored from
https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Benchmarks/CC-OCR/evaluation/evaluator
Only the multi_lan_ocr / multi_scene_ocr evaluators are needed.
"""
from .ocr_evaluator import OcrEvaluator
from .common import summary

evaluator_map_info = {
    "multi_lan_ocr": OcrEvaluator("multi_lan_ocr"),
    "multi_scene_ocr": OcrEvaluator("multi_scene_ocr"),
}
