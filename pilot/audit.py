"""Read-only curriculum audit of trusted checkpoints selected before evaluation."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import checkpoint
import environment  # noqa: F401 -- adds the pinned Foundation import path for pickle


def audit(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    records = {}
    for name, entry in manifest.items():
        source = Path(entry["path"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Selected checkpoint hash mismatch")
        saved = checkpoint.load(source)
        economies = []
        for blob in saved["workers"]:
            env = pickle.loads(blob)[0]
            tax = env.get_component("PeriodicBracketTax")
            economies.append(
                {
                    "episode_step": env.world.timestep,
                    "completions": env._completions,
                    "labor_weight": env.energy_weight,
                    "tax_ceiling": tax.curr_rate_max,
                    "tax_model": tax.tax_model,
                    "taxes_disabled": tax.disable_taxes,
                }
            )
        records[name] = {
            "progress": saved["progress"],
            "schedules": saved["schedules"],
            "phase2_from": saved["config"]["phase2_from"],
            "phase_one_progress": saved["config"]["phase_one_progress"],
            "planner_shapes": saved["config"]["planner_shapes"],
            "economies": economies,
        }
        if saved["config"]["phase2_from"] and any(
            e["labor_weight"] != 1 or (not e["taxes_disabled"] and e["tax_ceiling"] != 1)
            for e in economies
        ):
            raise ValueError("Selected phase-two checkpoint lacks full labor/tax support")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.output).open("x") as stream:
        json.dump(audit(args.manifest), stream, indent=2, allow_nan=False)
