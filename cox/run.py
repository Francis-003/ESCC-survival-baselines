"""Command-line entry point for the independent Cox baseline."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline_common import run_baseline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-root", type=Path, default=Path("try_output"))
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Compatibility argument; prepared model_table.tsv under output-root is authoritative.")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Compatibility argument; frozen CSV lists beside config are authoritative.")
    args = parser.parse_args()
    if args.data_root is not None or args.manifest is not None:
        parser.error("--data-root/--manifest cannot override prepared inputs; run prepare_data.py and ct_features.py first, then use --output-root")
    print(json.dumps(run_baseline("cox", args.config, args.output_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
