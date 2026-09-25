import json
import tempfile
import unittest
from pathlib import Path
from main import parse_args, resolve_runtime_config


def resolve(argv):
    parser, args = parse_args(["test-data"] + argv)
    return resolve_runtime_config(args, parser)


class LowMemoryCliTest(unittest.TestCase):
    def test_default(self):
        self.assertEqual(resolve([]).hypothesis_chunk, 64)

    def test_disable(self):
        self.assertEqual(resolve(["--hypothesis-chunk", "0"]).hypothesis_chunk, 0)

    def test_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "runtime.json"
            config.write_text(json.dumps({"hypothesis_chunk": 0, "seed": 19}))
            self.assertEqual(resolve(["--runtime-config", str(config)]).hypothesis_chunk, 0)
            c = resolve(["--runtime-config", str(config), "--hypothesis-chunk", "128"])
            self.assertEqual((c.hypothesis_chunk, c.seed), (128, 19))

    def test_negative(self):
        with self.assertRaises(SystemExit) as result:
            resolve(["--hypothesis-chunk", "-1"])
        self.assertEqual(result.exception.code, 2)
