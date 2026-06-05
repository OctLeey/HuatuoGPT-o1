import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clinical_rlvr.data import format_mcq_prompt, format_sft_response, normalize_sample
from clinical_rlvr.datasets import build_dpo_record, build_rlvr_record, build_sft_record
from clinical_rlvr.evaluation import classify_error, evaluate_records
from clinical_rlvr.generation import build_mock_output, build_mock_predictions
from clinical_rlvr.judge import build_judge_prompt, mock_judge, parse_judge_label
from clinical_rlvr.rewards import (
    answer_reward,
    composite_train_reward,
    entity_reward,
    extract_final_answer,
    format_reward,
)


class ClinicalRlvrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "data" / "demo_data.json").open(encoding="utf-8") as f:
            cls.raw_sample = json.load(f)[0]
        cls.sample = normalize_sample(cls.raw_sample, source="demo", idx=0)

    def test_normalize_sample_maps_original_fields(self):
        self.assertEqual(self.sample["answer_key"], "D")
        self.assertEqual(self.sample["source"], "demo")
        self.assertIn("Parvovirus", self.sample["answer"])
        self.assertIn("A", self.sample["options"])

    def test_prompt_and_sft_response_are_well_formed(self):
        prompt = format_mcq_prompt(self.sample, strict=True)
        response = format_sft_response(self.sample["rationale"], self.sample["answer_key"])

        self.assertIn(self.sample["question"], prompt)
        self.assertIn("A.", prompt)
        self.assertIn("<think>", response)
        self.assertIn("<answer>\nD\n</answer>", response)

    def test_answer_extraction_and_rewards(self):
        output = "<think>\nThe low reticulocyte count points to aplastic crisis.\n</think>\n<answer>\nD\n</answer>"
        wrong_output = "<think>\nThis suggests iron deficiency.\n</think>\n<answer>\nC\n</answer>"

        self.assertEqual(extract_final_answer(output, self.sample["options"]), "D")
        self.assertEqual(answer_reward(output, self.sample), 1.0)
        self.assertEqual(answer_reward(wrong_output, self.sample), 0.0)
        self.assertEqual(format_reward(output), 1.0)

    def test_entity_reward_and_composite_reward(self):
        output = (
            "<think>\nSevere anemia with low reticulocytes can occur during "
            "parvovirus B19 transient aplastic crisis.\n</think>\n<answer>\nD\n</answer>"
        )

        self.assertGreater(entity_reward(output, self.sample), 0.0)
        reward = composite_train_reward(output, self.sample)
        self.assertEqual(reward["answer"], 1.0)
        self.assertGreater(reward["total"], 1.0)

    def test_dataset_builders_create_training_formats(self):
        sft = build_sft_record(self.sample)
        dpo = build_dpo_record(self.sample)
        rlvr = build_rlvr_record(self.sample)

        self.assertEqual(sft["messages"][0]["role"], "user")
        self.assertIn("<answer>\nD\n</answer>", sft["response"])
        self.assertEqual(dpo["chosen_answer_key"], "D")
        self.assertNotEqual(dpo["chosen_answer_key"], dpo["rejected_answer_key"])
        self.assertEqual(rlvr["answer_key"], "D")
        self.assertIn("options", rlvr)

    def test_dataset_cli_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "clinical_rlvr_demo"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_clinical_rlvr_data.py"),
                    "--input",
                    str(ROOT / "data" / "demo_data.json"),
                    "--output_dir",
                    str(output_dir),
                    "--source",
                    "demo",
                    "--limit",
                    "3",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            for filename in ["normalized.json", "sft.json", "dpo.json", "rlvr.json", "summary.json"]:
                self.assertTrue((output_dir / filename).exists())

            with (output_dir / "summary.json").open(encoding="utf-8") as f:
                summary = json.load(f)
            self.assertEqual(summary["num_samples"], 3)

    def test_offline_evaluation_metrics_and_error_types(self):
        correct = dict(
            self.raw_sample,
            source="demo",
            output=(
                "<think>\nSevere anemia with low reticulocytes can occur during "
                "parvovirus B19 transient aplastic crisis.\n</think>\n<answer>\nD\n</answer>"
            ),
        )
        wrong = dict(
            self.raw_sample,
            source="demo",
            output="<think>\nThis looks like iron deficiency anemia.\n</think>\n<answer>\nC\n</answer>",
        )
        invalid = dict(self.raw_sample, source="demo", output="I cannot determine the answer.")

        report = evaluate_records([correct, wrong, invalid])

        self.assertEqual(report["num_samples"], 3)
        self.assertAlmostEqual(report["accuracy"], 1 / 3)
        self.assertIn("correct", report["error_types"])
        self.assertIn("invalid_output_format", report["error_types"])
        self.assertEqual(classify_error(self.sample, invalid["output"]), "invalid_output_format")

    def test_evaluation_cli_writes_report(self):
        records = [
            dict(
                self.raw_sample,
                source="demo",
                output="<think>\nParvovirus B19 explains the low reticulocytes.\n</think>\n<answer>\nD\n</answer>",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "predictions.json"
            report_path = Path(tmpdir) / "report.json"
            with input_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_clinical_rlvr_outputs.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(report_path),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            with report_path.open(encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["num_samples"], 1)
            self.assertEqual(report["accuracy"], 1.0)

    def test_sft_qlora_dry_run_validates_config_and_data(self):
        from train.sft_qlora import dry_run

        datasets = {
            "train": [
                {
                    "prompt": "Please answer.\nA. x\nB. y",
                    "response": "<think>\nReason.\n</think>\n<answer>\nA\n</answer>",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = Path(tmpdir) / "sft.json"
            with train_file.open("w", encoding="utf-8") as f:
                json.dump(datasets["train"], f)

            summary = dry_run(
                {
                    "model_name_or_path": "Qwen/Qwen3-4B",
                    "train_file": str(train_file),
                    "output_dir": str(Path(tmpdir) / "ckpt"),
                    "max_seq_length": 1024,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "learning_rate": 0.0002,
                }
            )

            self.assertEqual(summary["num_records"], 1)
            self.assertEqual(summary["model_name_or_path"], "Qwen/Qwen3-4B")

    def test_mock_generation_builds_evaluable_predictions(self):
        rlvr_record = build_rlvr_record(self.sample)
        output = build_mock_output(rlvr_record)
        predictions = build_mock_predictions([rlvr_record])
        report = evaluate_records(predictions)

        self.assertIn("<answer>\nD\n</answer>", output)
        self.assertEqual(predictions[0]["generation_mode"], "mock_correct")
        self.assertEqual(report["accuracy"], 1.0)

    def test_generation_cli_mock_outputs_feed_evaluator(self):
        rlvr_record = build_rlvr_record(self.sample)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "rlvr.json"
            predictions_path = Path(tmpdir) / "predictions.json"
            report_path = Path(tmpdir) / "report.json"

            with input_path.open("w", encoding="utf-8") as f:
                json.dump([rlvr_record], f, ensure_ascii=False)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "generate_clinical_rlvr_outputs.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(predictions_path),
                    "--mock",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_clinical_rlvr_outputs.py"),
                    "--input",
                    str(predictions_path),
                    "--output",
                    str(report_path),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            with report_path.open(encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["num_samples"], 1)
            self.assertEqual(report["accuracy"], 1.0)

    def test_grpo_dry_run_validates_rewards(self):
        from train.grpo_rlvr import dry_run

        rlvr_record = build_rlvr_record(self.sample)
        with tempfile.TemporaryDirectory() as tmpdir:
            train_file = Path(tmpdir) / "rlvr.json"
            with train_file.open("w", encoding="utf-8") as f:
                json.dump([rlvr_record], f, ensure_ascii=False)

            summary = dry_run(
                {
                    "model_name_or_path": "Qwen/Qwen3-4B",
                    "train_file": str(train_file),
                    "output_dir": str(Path(tmpdir) / "grpo_ckpt"),
                    "max_prompt_length": 2048,
                    "max_completion_length": 1024,
                    "num_generations": 4,
                    "per_device_train_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "learning_rate": 0.000001,
                    "beta": 0.04,
                    "answer_reward_weight": 1.0,
                    "format_reward_weight": 0.2,
                    "entity_reward_weight": 0.3,
                    "length_penalty_weight": -0.1,
                }
            )

            self.assertEqual(summary["checked_records"], 1)
            self.assertIn("clinical_answer_reward", summary["reward_breakdown"])
            self.assertGreater(summary["reward_totals"][0], 1.0)

    def test_judge_helpers_parse_and_score_mock_outputs(self):
        rlvr_record = build_rlvr_record(self.sample)
        prediction = dict(rlvr_record, model_output=build_mock_output(rlvr_record))

        self.assertEqual(parse_judge_label("correct"), "correct")
        self.assertEqual(parse_judge_label("Incorrect because the diagnosis is wrong."), "incorrect")
        self.assertEqual(parse_judge_label("not sure"), "uncertain")
        self.assertIn("Reference Answer", build_judge_prompt(prediction))

        judged = mock_judge(prediction)
        self.assertEqual(judged["judge_label"], "correct")
        self.assertEqual(judged["judge_score"], 1.0)

    def test_judge_cli_mock_outputs_feed_evaluator(self):
        rlvr_record = build_rlvr_record(self.sample)
        prediction = dict(rlvr_record, model_output=build_mock_output(rlvr_record))
        with tempfile.TemporaryDirectory() as tmpdir:
            predictions_path = Path(tmpdir) / "predictions.json"
            judged_path = Path(tmpdir) / "judged.json"
            report_path = Path(tmpdir) / "report.json"
            cache_path = Path(tmpdir) / "judge_cache.json"

            with predictions_path.open("w", encoding="utf-8") as f:
                json.dump([prediction], f, ensure_ascii=False)

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "judge_clinical_rlvr_outputs.py"),
                    "--input",
                    str(predictions_path),
                    "--output",
                    str(judged_path),
                    "--cache",
                    str(cache_path),
                    "--mock",
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_clinical_rlvr_outputs.py"),
                    "--input",
                    str(judged_path),
                    "--output",
                    str(report_path),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            with report_path.open(encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(report["judge_pass_rate"], 1.0)
            self.assertEqual(report["judge_labels"], {"correct": 1})


if __name__ == "__main__":
    unittest.main()
