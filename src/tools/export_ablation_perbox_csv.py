"""
Exports per-box (not just aggregate) real-world evaluation results across the
three model sizes tested during ablation (Nano/Medium/Large), all under the
production single-stage/no-HMM configuration. One row per localized
ground-truth box, with the model, image, true class, predicted class,
correctness, and IoU -- enough to run real statistical tests (confidence
intervals, paired significance tests, per-image variance) rather than only
comparing point-estimate accuracy numbers.

Run: python -m src.tools.export_ablation_perbox_csv
Output: analysis/ablation_perbox_results.csv
"""
import csv
import pathlib
import shutil

from src.model.pipeline import PanelSafePipeline
from src.tools.evaluate_pipeline import evaluate_on_real_dataset

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "src" / "model" / "pipeline_config.json"

MODEL_WEIGHTS = {
    "Nano": MODELS_DIR / "best_round2_nano_baseline.pt",
    "Medium": MODELS_DIR / "best_round2_medium_baseline.pt",
    "Large": pathlib.Path(
        r"C:\Users\PC GAMING\breaker-detection-project\runs\detect\model\runs\train-20\weights\best.pt"
    ),
}


def main():
    active_model_path = MODELS_DIR / "best.pt"
    original_config = CONFIG_PATH.read_text(encoding="utf-8")
    original_active_weights = active_model_path.read_bytes()

    all_rows = []
    try:
        for model_name, weights_path in MODEL_WEIGHTS.items():
            if not weights_path.exists():
                print(f"Skipping {model_name}: weights not found at {weights_path}")
                continue

            print(f"\n=== Evaluating {model_name} ({weights_path.name}) ===")
            shutil.copy2(weights_path, active_model_path)

            pipeline = PanelSafePipeline(config_path=str(CONFIG_PATH))
            pipeline.config["classifier_mode"] = "single_stage"
            pipeline.config["use_hmm"] = False

            metrics = evaluate_on_real_dataset(pipeline)
            if not metrics:
                continue

            print(f"  localization_acc={metrics['localization_acc']:.2%}  classification_acc={metrics['classification_acc']:.2%}")

            for record in metrics["per_box_records"]:
                row = dict(record)
                row["model"] = model_name
                all_rows.append(row)
    finally:
        # Restore whatever was actually running before this script touched it
        active_model_path.write_bytes(original_active_weights)
        CONFIG_PATH.write_text(original_config, encoding="utf-8")
        print("\nRestored original models/best.pt and pipeline_config.json.")

    out_path = PROJECT_ROOT / "analysis" / "ablation_perbox_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "image", "true_class", "pred_class", "correct", "iou"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} per-box rows to {out_path}")


if __name__ == "__main__":
    main()
