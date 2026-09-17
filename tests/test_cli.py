"""CLI 解析测试：默认值、开关语义、交错参数形式与错误退出码。"""

import contextlib
import io
import unittest

import main


EXPECTED_DEFAULTS = {
    "chain_threshold": 0.45,
    "initial_domain_threshold": 0.45,
    "min_domain_threshold": 0.35,
    "similarity_threshold": 0.85,
    "complex_threshold": 0.35,
    "complex_domain_opt": False,
    "improve_accepted": True,
    "domain_opt": True,
    "save_all_attempts": False,
    "cleanup": False,
    "do_refine": True,
    "complex_min_cc": 0.25,
    "refine_tm": 0.75,
    "num_processes": 8,
    "batch_size": 8,
    "homo_chain_refine": False,
    "pre_screen": True,
    "pre_screen_by_domain": False,
    "no_domain_split_chains": frozenset(),
}


class ParseArgsTest(unittest.TestCase):

    def parse(self, argv):
        parser, args = main.parse_args(argv)
        self.assertIsNotNone(args)
        return args

    def exit_code(self, argv):
        with self.assertRaises(SystemExit) as ctx:
            with contextlib.redirect_stderr(io.StringIO()):
                main.parse_args(argv)
        return ctx.exception.code

    def test_defaults_match_documented_values(self):
        args = self.parse(["/data/case"])
        self.assertEqual(main.assembly_kwargs_from(args), EXPECTED_DEFAULTS)

    def test_switches_enable(self):
        args = self.parse(["/data/case", "--complex-domain-opt",
                           "--save-all-attempts", "--cleanup",
                           "--homo-chain-refine", "--pre-screen-by-domain"])
        kwargs = main.assembly_kwargs_from(args)
        self.assertTrue(kwargs["complex_domain_opt"])
        self.assertTrue(kwargs["save_all_attempts"])
        self.assertTrue(kwargs["cleanup"])
        self.assertTrue(kwargs["homo_chain_refine"])
        self.assertTrue(kwargs["pre_screen_by_domain"])

    def test_switches_disable(self):
        args = self.parse(["/data/case", "--no-improve-accepted",
                           "--no-domain-opt", "--no-refine", "--no-pre-screen"])
        kwargs = main.assembly_kwargs_from(args)
        self.assertFalse(kwargs["improve_accepted"])
        self.assertFalse(kwargs["domain_opt"])
        self.assertFalse(kwargs["do_refine"])
        self.assertFalse(kwargs["pre_screen"])

    def test_numeric_options_override_defaults(self):
        args = self.parse(["/data/case", "--chain-threshold", "0.4",
                           "--domain-threshold", "0.41", "--domain-min-cc", "0.3",
                           "--complex-threshold", "0.33", "--complex-min-cc", "0.2",
                           "--similarity-threshold", "0.8", "--refine-tm", "0.7",
                           "--num-processes", "16", "--batch-size", "4"])
        kwargs = main.assembly_kwargs_from(args)
        self.assertEqual(kwargs["chain_threshold"], 0.4)
        self.assertEqual(kwargs["initial_domain_threshold"], 0.41)
        self.assertEqual(kwargs["min_domain_threshold"], 0.3)
        self.assertEqual(kwargs["complex_threshold"], 0.33)
        self.assertEqual(kwargs["complex_min_cc"], 0.2)
        self.assertEqual(kwargs["similarity_threshold"], 0.8)
        self.assertEqual(kwargs["refine_tm"], 0.7)
        self.assertEqual(kwargs["num_processes"], 16)
        self.assertEqual(kwargs["batch_size"], 4)

    def test_no_domain_split_parses_comma_list(self):
        args = self.parse(["/data/case", "--no-domain-split", " A , B ,"])
        self.assertEqual(main.assembly_kwargs_from(args)["no_domain_split_chains"],
                         frozenset({"A", "B"}))

    def test_manual_mode_positionals(self):
        args = self.parse(["a.mrc", "structs", "3.5", "0.02", "out"])
        self.assertEqual(args.paths, ["a.mrc", "structs", "3.5", "0.02", "out"])

    def test_option_between_positionals(self):
        args = self.parse(["a.mrc", "structs", "--log", "3.5", "0.02"])
        self.assertEqual(args.paths, ["a.mrc", "structs", "3.5", "0.02"])
        self.assertTrue(args.log)

    def test_option_with_value_between_positionals(self):
        args = self.parse(["a", "--num-processes", "4", "b", "c", "d"])
        self.assertEqual(args.paths, ["a", "b", "c", "d"])
        self.assertEqual(args.num_processes, 4)

    def test_option_before_positional(self):
        args = self.parse(["--log", "/data/case"])
        self.assertEqual(args.paths, ["/data/case"])
        self.assertTrue(args.log)

    def test_log_file_overrides_log(self):
        args = self.parse(["/data/case", "--log", "--log-file", "/tmp/x.log"])
        self.assertEqual(main.resolve_log_file(args), "/tmp/x.log")
        args = self.parse(["/data/case", "--log"])
        self.assertIs(main.resolve_log_file(args), True)
        args = self.parse(["/data/case"])
        self.assertIsNone(main.resolve_log_file(args))

    def test_no_positional_prints_help_and_returns_none(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            parser, args = main.parse_args([])
        self.assertIsNone(args)
        self.assertIn("usage", out.getvalue())

    def test_wrong_positional_count_exits_2(self):
        for argv in (["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d", "e", "f"]):
            self.assertEqual(self.exit_code(argv), 2, argv)

    def test_unknown_option_exits_2(self):
        self.assertEqual(self.exit_code(["/data/case", "--nope"]), 2)

    def test_missing_value_exits_2(self):
        self.assertEqual(self.exit_code(["/data/case", "--chain-threshold"]), 2)

    def test_non_numeric_value_exits_2(self):
        self.assertEqual(self.exit_code(["/data/case", "--chain-threshold", "abc"]), 2)


if __name__ == "__main__":
    unittest.main()
