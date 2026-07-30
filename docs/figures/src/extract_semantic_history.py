"""Extract per-epoch history from a W&B offline run into data/semantic_history.json.

Usage (repo root, project venv — needs the wandb package):
    .venv/Scripts/python docs/figures/src/extract_semantic_history.py wandb/offline-run-<id>
"""
import json
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore


def main() -> int:
    run_dir = Path(sys.argv[1])
    wandb_files = list(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        print(f"No .wandb file in {run_dir}")
        return 1

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))

    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for item in rec.history.item:
                key = item.key or "/".join(item.nested_key)
                try:
                    row[key] = json.loads(item.value_json)
                except (json.JSONDecodeError, ValueError):
                    pass
            rows.append(row)

    out = Path(__file__).resolve().parents[1] / "data" / "semantic_history.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} history rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
