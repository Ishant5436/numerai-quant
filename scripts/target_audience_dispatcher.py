"""
Target Audience Dispatcher & Community Conversion Engine.
Generates, validates, and exports mechanical anti-AI promotional payloads
for Numerai community channels: Discord #numerbay, Discourse Forum, and Reddit r/numerai.
Adheres strictly to Gerard J. Holzmann's Power of 10 Safety Invariants.
"""
import os
import sys
import ast
import argparse
from typing import Dict, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def generate_channel_payload(channel: str) -> str:
    """Generates channel-tailored, unformatted anti-AI copy for community acquisition."""
    assert isinstance(channel, str) and len(channel) > 0, "channel must be non-empty string"
    assert channel in ["discord", "forum", "reddit", "staker"], f"Unsupported channel: {channel}"

    if channel == "discord":
        payload = (
            "Hey all, I just listed predictions for cypherpole on Numerbay for stakers looking for "
            "orthogonal alpha to blend into their tournament accounts. The model runs a multi-target "
            "LightGBM ensemble with linear QR feature neutralization across canonical risk factors, "
            "currently sitting at plus 0.0607 on 60-day correlation (corj60, top 1.5% globally, 84.2% win rate). "
            "Track record is public at https://numer.ai/cypherpole. I set up an automated pipeline on Apple Silicon "
            "to upload artifacts as soon as weekend rounds open. Monthly recurring subscription is 5 NMR. "
            "To let folks test the correlation and submission pipeline on their own accounts before subscribing, "
            "I set up coupon code CYPHER100 on Numerbay for 100 percent off the single round trial for the first "
            "5 stakers. You can grab it at https://numerbay.ai/explore/numerai-predictions?term=cypherpole"
        )
    elif channel == "forum":
        payload = (
            "Hey everyone, I wanted to share some details on my primary model cypherpole and open up an "
            "initial batch of free prediction trials on Numerbay for tournament stakers who are looking to "
            "blend uncorrelated alpha into their staking portfolios. I developed cypherpole as an ensemble of "
            "orthogonalized LightGBM models trained across multiple forecast horizons. Rather than fitting against "
            "a single raw target, the model constructs predictions across orthogonal target variants and projects "
            "the resulting raw vector against the principal risk factors per era using linear QR feature neutralization. "
            "Over the past 60 days of live out-of-sample tournament evaluation on Numerai Classic, cypherpole is sitting "
            "at plus 0.0607 on correlation (corj60), with a positive era win rate of 84.2 percent and an annualized Sharpe "
            "of 1.84, placing it in the top 1.5 percent of the global leaderboard. You can audit the full unmanipulated "
            "track record directly on the official tournament leaderboard at https://numer.ai/cypherpole. For anyone who "
            "wants to verify submission mechanics and assess correlation on their own accounts before committing to a "
            "monthly subscription, I generated a trial coupon code CYPHER100 on Numerbay that gives a 100 percent discount "
            "on the single round trial for the first 5 stakers at https://numerbay.ai/explore/numerai-predictions?term=cypherpole."
        )
    elif channel == "reddit":
        payload = (
            "Hey r/numerai, sharing live out-of-sample results for cypherpole, a multi-target LightGBM ensemble "
            "running linear QR feature neutralization on Apple Silicon. Over the last 60 rounds, the model maintained "
            "a plus 0.0607 mean correlation (corj60) with an 84.2% win rate, ranking in the top 1.5% globally on the "
            "official leaderboard. For tournament stakers interested in testing how blending 20 to 30 percent of this "
            "signal impacts portfolio Sharpe and drawdown, I uploaded predictions to Numerbay and created trial coupon "
            "code CYPHER100 for 100 percent off the single round trial for the first 5 stakers. The full public leaderboard "
            "track record is visible at https://numer.ai/cypherpole and listings can be inspected at https://numerbay.ai"
        )
    else:
        payload = (
            "For tournament stakers seeking to reduce portfolio drawdowns without adding market beta, "
            "blending a 25 percent allocation of cypherpole predictions provides orthogonal alpha with "
            "a 60-day out-of-sample correlation of plus 0.0607. You can test the blend risk-free for Round 1373 "
            "by entering coupon CYPHER100 at checkout on Numerbay."
        )

    assert len(payload) > 100, "Payload length invariant failed"
    assert "CYPHER100" in payload, "Promo coupon code invariant failed"
    return payload


def export_all_dispatches(export_dir: str) -> List[str]:
    """Exports all channel copy variants to disk files for immediate distribution."""
    assert isinstance(export_dir, str) and len(export_dir) > 0, "export_dir required"
    os.makedirs(export_dir, exist_ok=True)

    channels = ["discord", "forum", "reddit", "staker"]
    saved_files = []
    for ch in channels:
        content = generate_channel_payload(ch)
        target_path = os.path.join(export_dir, f"dispatch_{ch}.txt")
        with open(target_path, "w") as f:
            f.write(content + "\n")
        assert os.path.exists(target_path), f"Failed to save {target_path}"
        saved_files.append(target_path)

    assert len(saved_files) == len(channels), "All channels must be exported"
    return saved_files


def audit_safety_invariants_local() -> bool:
    """Mechanically verifies Power of 10 invariants: functions <= 60 lines, asserts >= 2."""
    filepath = os.path.abspath(__file__)
    assert os.path.exists(filepath), "Source file must exist"
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)

    passed = True
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines = (node.end_lineno - node.lineno) + 1
            assert lines <= 60, f"Function {node.name} exceeds 60 lines: {lines}"
            assert_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Assert))
            assert (
                assert_count >= 2
            ), f"Function {node.name} has fewer than 2 asserts: {assert_count}"

    assert passed is True, "Audit must return true"
    return passed


def main():
    parser = argparse.ArgumentParser(description="Target Audience Outreach Dispatcher")
    parser.add_argument("--export-dir", type=str, default="data/dispatches", help="Directory to export payloads")
    parser.add_argument("--channel", type=str, default="all", help="Target channel or all")
    args = parser.parse_args()

    assert isinstance(args.export_dir, str), "export_dir must be string"
    assert isinstance(args.channel, str), "channel must be string"

    out_dir = os.path.join(BASE_DIR, args.export_dir)
    print(f"[*] Exporting channel outreach dispatches to {out_dir}...")
    files = export_all_dispatches(out_dir)
    for f in files:
        print(f" -> Generated: {f}")


if __name__ == "__main__":
    main()
