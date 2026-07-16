import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from scripts.train_ppo_agent import (  # noqa: E402
    build_experiment_name,
    resolve_training_output_dir,
)


class TrainPPOAgentScriptTests(unittest.TestCase):
    def test_experiment_name_for_single_depth(self):
        name = build_experiment_name(
            min_depth=1,
            max_depth=1,
            observation_encoding="one_hot",
        )

        self.assertEqual(name, "depth_1_onehot")

    def test_experiment_name_for_depth_range(self):
        name = build_experiment_name(
            min_depth=1,
            max_depth=2,
            observation_encoding="normalized",
        )

        self.assertEqual(name, "depth_1_2_normalized")

    def test_experiment_name_includes_optional_warm_start_suffix(self):
        name = build_experiment_name(
            min_depth=1,
            max_depth=10,
            observation_encoding="one_hot",
            experiment_suffix="warm_frontier_1_6",
        )

        self.assertEqual(name, "depth_1_10_onehot_warm_frontier_1_6")

    def test_first_auto_version_resolves_to_v001(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir, metadata = resolve_training_output_dir(
                output_dir=None,
                output_root=Path(directory),
                min_depth=1,
                max_depth=1,
                observation_encoding="one_hot",
                version=None,
                overwrite=False,
            )

        self.assertEqual(output_dir.name, "v001")
        self.assertEqual(output_dir.parent.name, "depth_1_onehot")
        self.assertEqual(metadata["version"], "v001")
        self.assertEqual(metadata["output_mode"], "versioned")

    def test_next_auto_version_resolves_to_v002(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "depth_1_onehot" / "v001").mkdir(parents=True)

            output_dir, metadata = resolve_training_output_dir(
                output_dir=None,
                output_root=root,
                min_depth=1,
                max_depth=1,
                observation_encoding="one_hot",
                version=None,
                overwrite=False,
            )

        self.assertEqual(output_dir.name, "v002")
        self.assertEqual(metadata["version"], "v002")

    def test_overwrite_reuses_latest_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "depth_1_onehot" / "v001").mkdir(parents=True)
            (root / "depth_1_onehot" / "v002").mkdir(parents=True)

            output_dir, metadata = resolve_training_output_dir(
                output_dir=None,
                output_root=root,
                min_depth=1,
                max_depth=1,
                observation_encoding="one_hot",
                version=None,
                overwrite=True,
            )

        self.assertEqual(output_dir.name, "v002")
        self.assertEqual(metadata["version"], "v002")
        self.assertTrue(metadata["overwrite"])

    def test_explicit_version_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir, metadata = resolve_training_output_dir(
                output_dir=None,
                output_root=Path(directory),
                min_depth=2,
                max_depth=2,
                observation_encoding="normalized",
                version="lr1e-4",
                overwrite=False,
            )

        self.assertEqual(output_dir.name, "lr1e-4")
        self.assertEqual(output_dir.parent.name, "depth_2_normalized")
        self.assertEqual(metadata["version"], "lr1e-4")

    def test_explicit_output_dir_bypasses_auto_versioning(self):
        with tempfile.TemporaryDirectory() as directory:
            explicit_path = Path(directory) / "custom"
            output_dir, metadata = resolve_training_output_dir(
                output_dir=explicit_path,
                output_root=Path(directory) / "ignored",
                min_depth=1,
                max_depth=5,
                observation_encoding="one_hot",
                version="v999",
                overwrite=True,
            )

        self.assertEqual(output_dir, explicit_path)
        self.assertEqual(metadata["output_mode"], "explicit")
        self.assertIsNone(metadata["experiment_name"])
        self.assertIsNone(metadata["version"])


if __name__ == "__main__":
    unittest.main()
