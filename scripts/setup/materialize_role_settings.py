"""Resolve tracked role templates to checkout-specific absolute permission paths."""
from pathlib import Path
import argparse, json
ROOT = Path(__file__).resolve().parents[2]
def materialize(role):
    if role not in {"implementer", "verifier", "scribe"}:
        raise ValueError(role)
    data = json.loads((ROOT / f".claude/roles/{role}.settings.json").read_text())
    for kind, rules in data.get("permissions", {}).items():
        for index, rule in enumerate(rules):
            tool, separator, tail = rule.partition("(/")
            if separator and not tail.startswith("/"):
                rules[index] = tool + "(/" + ROOT.as_posix() + "/" + tail
        assert all("${" not in rule for rule in rules)
    target = ROOT / f".claude/roles/{role}.local.settings.json"
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return target
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["implementer", "verifier", "scribe"])
    print(materialize(parser.parse_args().role))
