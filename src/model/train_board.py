"""Train the 1-class BOARD detector used by the phone-side viewfinder.

This is NOT the audit model. Its only job is to answer "where is the board in
this preview frame, and how much of the frame does it fill", so the app can tell
the user to move closer / hold still and auto-capture when the shot is good. The
6-class audit still runs server-side on the full-resolution photo.

Choices that differ from train.py, and why:

- **Nano, not Medium/Large.** This has to run in a browser on a phone, on a
  downscaled preview stream. Capacity is not the constraint; a board is a huge,
  high-contrast target.
- **imgsz=320, not 1280.** The audit model needs 1280 to resolve the lettering
  on an 18mm module. The viewfinder resolves one object that fills roughly 44%
  of the frame (measured across the dataset by prepare_board_dataset.py), so
  320 is plenty and keeps inference inside a live frame budget.
- **No copy_paste.** It pastes instances between images, which is meaningless
  with one class covering half the frame, and it would fabricate images with two
  boards in them.
- **No mixup.** It was measured to hurt on this project's real photos
  (docs/okf/methodology/ablation_study.md) and blending two boards is exactly
  the wrong training signal for "is one board framed well".
- **Wide scale jitter, kept.** The model must recognise a board at 5% and at 80%
  of the frame, since telling those apart is the whole feature.

The starting weights are the stock Ultralytics COCO checkpoint. That is the
correct use of that file (pretrained backbone for fine-tuning); it is only wrong
when someone mistakes it for a trained PanelSafe model.
"""
import argparse
import os
from pathlib import Path

import mlflow
from ultralytics import YOLO

# Absolute so every experiment lands in the one tracking store, whichever
# worktree this is run from.
TRACKING_URI = "sqlite:///C:/ironhack/labs/breaker-detection-project/mlflow.db"
PRETRAINED = r"C:\ironhack\labs\breaker-detection-project\yolo26n.pt"


def setup_tracking(fallback_dir):
    """Point MLflow at a store that actually works, and pin it for ultralytics.

    Two traps, both hit on the first run of this script:

    1. `mlflow.db` was created by the MLflow pinned during Phase 2. The installed
       client rejects its schema until someone runs `mlflow db upgrade` on it.
       That is a backup-first decision about an existing experiment store, not
       something a training script gets to make, so this falls back to a
       self-contained file store beside the run instead of touching it.
    2. Ultralytics registers its own MLflow callback, which re-resolves the URI
       at `on_pretrain_routine_end` and raises there -- outside any try/except in
       this file -- killing training after setup "succeeded". Exporting
       MLFLOW_TRACKING_URI is what that callback reads, so pinning it here keeps
       both halves pointed at the same working store.
    """
    experiment = "PanelSafe-Board-Viewfinder"
    # A *database* backend, not a directory: this MLflow refuses file stores
    # ("maintenance mode") unless MLFLOW_ALLOW_FILE_STORE=true, so the obvious
    # "just log next to the run" fallback fails too. A fresh .db file gets
    # created with the current schema, which is what the legacy mlflow.db lacks.
    fallback_uri = "sqlite:///" + str(
        (Path(fallback_dir).absolute() / "mlflow_board.db")).replace("\\", "/")
    for uri in (os.environ.get("MLFLOW_TRACKING_URI") or TRACKING_URI, fallback_uri):
        try:
            Path(fallback_dir).mkdir(parents=True, exist_ok=True)
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            mlflow.autolog()
        except Exception as err:
            print(f"WARNING: MLflow store unusable at {uri}\n"
                  f"         {err.__class__.__name__}: {str(err).splitlines()[0]}")
            continue
        os.environ["MLFLOW_TRACKING_URI"] = uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
        print(f"MLflow tracking -> {uri}")
        return uri
    return None


def run_training(args):
    tracking_uri = setup_tracking(os.path.join("model", "runs", "mlflow"))

    model = YOLO(args.weights)

    if not tracking_uri:
        # Ultralytics' own MLflow callback re-resolves the URI mid-training and
        # would raise there, outside any try/except here. Strip it from this
        # model instance only -- `yolo settings mlflow=False` would disable
        # tracking globally, including for the 6-class audit model.
        removed = 0
        for event, callbacks in model.callbacks.items():
            kept = [cb for cb in callbacks if "mlflow" not in getattr(cb, "__module__", "")]
            removed += len(callbacks) - len(kept)
            model.callbacks[event] = kept
        print(f"Stripped {removed} ultralytics MLflow callbacks; run is untracked "
              f"(weights and results.csv still land in model/runs/).")

    print(f"Training BOARD detector: {args.weights} -> {args.data} "
          f"@ {args.imgsz}px, {args.epochs} epochs, batch {args.batch}")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,             # Windows: dataloader workers are a known hang source here
        project="model/runs",
        name=args.name,
        optimizer="auto",
        patience=15,
        mosaic=1.0,            # 4-image composites: varied framing/context
        close_mosaic=10,       # ...but the last 10 epochs see real, whole frames
        scale=0.5,             # 0.5x-1.5x: the fill-fraction range the app must judge
        degrees=5.0,           # phones are held slightly rotated
        hsv_v=0.5,             # basements, ceiling lights, phone flash
        copy_paste=0.0,
        mixup=0.0,
    )

    metrics = model.val(data=args.data, imgsz=args.imgsz, device=0)
    print("\n=== validation ===")
    print(f"mAP50     {metrics.box.map50:.4f}")
    print(f"mAP50-95  {metrics.box.map:.4f}")
    print(f"precision {metrics.box.mp:.4f}")
    print(f"recall    {metrics.box.mr:.4f}")
    # Ask the trainer where it actually saved. `project` is resolved against
    # ultralytics' own runs_dir setting, not the working directory, so composing
    # the path by hand prints a path that does not exist.
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    print("\nWeights:", os.path.join(str(save_dir), "weights", "best.pt") if save_dir
          else "see 'Results saved to' above")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_board.yaml")
    ap.add_argument("--weights", default=PRETRAINED)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="board-viewfinder")
    run_training(ap.parse_args())
