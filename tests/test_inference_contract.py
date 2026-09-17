"""T03 推理契约测试：输出字段白名单、无 release_cuda / 逐对 empty_cache、推理在 no_grad 下。

这些契约静态可查，用 AST 检查代码本身，避免"改回去没人发现"：
  - demo_mask 只读取 INFERENCE_OUTPUT_FIELDS 里的模型输出字段；
  - demo_mask 不再出现 release_cuda 递归释放与 torch.cuda.empty_cache()（T03 移除的工作）；
  - process_single_pair 被 torch.no_grad() 装饰（推理不建计算图）。
"""

import ast
import os
import unittest

from protassem.fitting.parenet.model import (
    INFERENCE_OUTPUT_FIELDS,
    select_output_fields,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_MASK_PATH = os.path.join(PROJECT_ROOT, "protassem", "fitting", "demo_mask.py")

# 推理侧真实消费的模型输出字段（demo_mask.process_single_pair）
CONSUMED_FIELDS = ("estimated_transform", "ref_points", "src_points")


def parse_demo_mask():
    with open(DEMO_MASK_PATH, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=DEMO_MASK_PATH)


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("function %s not found" % name)


def subscript_key(node):
    """取 output_dict['x'] 的字符串键；Python 3.8 的下标是 ast.Index 包装的。"""
    target = node.slice
    if isinstance(target, getattr(ast, "Index", ())):   # Python 3.8
        target = target.value
    if isinstance(target, ast.Constant) and isinstance(target.value, str):
        return target.value
    return None


def output_dict_subscripts(node):
    """收集 output_dict['字段'] 形式的读取键。"""
    keys = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Subscript):
            continue
        value = item.value
        if isinstance(value, ast.Name) and value.id == "output_dict":
            key = subscript_key(item)
            if key is not None:
                keys.add(key)
    return keys


def decorator_name(node):
    """把装饰器节点还原成名字（Python 3.8 没有 ast.unparse）。"""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


class OutputFieldContractTest(unittest.TestCase):
    def test_whitelist_matches_consumers(self):
        self.assertEqual(tuple(CONSUMED_FIELDS), tuple(INFERENCE_OUTPUT_FIELDS))

    def test_process_single_pair_only_reads_whitelisted_fields(self):
        keys = output_dict_subscripts(find_function(parse_demo_mask(), "process_single_pair"))
        self.assertEqual(set(INFERENCE_OUTPUT_FIELDS), keys)

    def test_select_keeps_everything_when_none(self):
        original = {"estimated_transform": 1, "hypotheses": 2, "ref_points": 3}
        self.assertIs(select_output_fields(original, None), original)

    def test_select_filters_and_preserves_order(self):
        original = {"a": 1, "b": 2, "c": 3}
        self.assertEqual({"c": 3, "a": 1}, select_output_fields(original, ("c", "a")))

    def test_select_raises_on_missing_field(self):
        with self.assertRaises(KeyError):
            select_output_fields({"estimated_transform": 1}, ("estimated_transform", "hypotheses"))


class InferenceMemoryContractTest(unittest.TestCase):
    def setUp(self):
        self.tree = parse_demo_mask()
        self.function = find_function(self.tree, "process_single_pair")

    def test_no_recursive_release_cuda(self):
        called = {node.func.id for node in ast.walk(self.function)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertNotIn("release_cuda", called)

    def test_no_empty_cache_in_inference_path(self):
        attrs = [node.func.attr for node in ast.walk(self.function)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("empty_cache", attrs)

    def test_function_has_no_grad_decorator(self):
        decorators = [decorator_name(item) for item in self.function.decorator_list]
        self.assertIn("no_grad", decorators,
                      "process_single_pair 必须由 torch.no_grad() 覆盖：%s" % decorators)

    def test_model_is_called_with_output_fields(self):
        calls = [node for node in ast.walk(self.function)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id == "model"]
        self.assertEqual(1, len(calls))
        keywords = {keyword.arg for keyword in calls[0].keywords}
        self.assertIn("output_fields", keywords)

    def test_demo_mask_does_not_import_release_cuda(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                imported.update(alias.name for alias in node.names)
        self.assertNotIn("release_cuda", imported)


if __name__ == "__main__":
    unittest.main()
