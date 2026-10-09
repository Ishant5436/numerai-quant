"""
Target Audience Channel Targeting & Distribution Registry.
Maps verified channels where Numerai tournament stakers and NMR capital allocators
congregate to maximize prediction subscription conversion and trial redemptions.
Deterministic Safety-Critical Standards: Bounded loops, assertions, <=60 line functions.
"""
import os
import sys
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def get_channel_registry() -> List[Dict[str, Any]]:
    """Returns canonical database of high-yield audience channels for Numerai stakers."""
    channels = [
        {
            "name": "Numerai Official Tournament Leaderboard",
            "platform": "Numerai Tournament Web",
            "url": "https://numer.ai/cypherpole",
            "audience": "Active tournament quants auditing top 50 models",
            "intent_level": "CRITICAL",
            "status": "LIVE & ACTIVE",
            "actionable_copy_file": "scripts/sync_leaderboard_billboard.py",
            "strategy": "Inbound 1-click button on leaderboard profile directing to Numerbay listing",
        },
        {
            "name": "Numerai Discord (#numerbay channel)",
            "platform": "Discord",
            "url": "https://discord.gg/numerai",
            "audience": "15,000+ members; dedicated marketplace channel for buyers/sellers",
            "intent_level": "CRITICAL",
            "status": "READY TO POST",
            "actionable_copy_file": "docs/DISCORD_ANNOUNCEMENT.md",
            "strategy": "Post unformatted solo developer announcement with CYPHER100 free coupon",
        },
        {
            "name": "Numerai Community Forum (Community Projects / Numerbay)",
            "platform": "Discourse Forum",
            "url": "https://forum.numer.ai/c/community-projects/14",
            "audience": "High-capital stakers, Council of Elders, long-term allocators",
            "intent_level": "HIGH",
            "status": "READY TO POST",
            "actionable_copy_file": "docs/NUMERAI_FORUM_PITCH.md",
            "strategy": "Reply to official '[Updates] NumerBay' thread or create Model Share case study",
        },
        {
            "name": "Reddit (r/numerai Community)",
            "platform": "Reddit",
            "url": "https://www.reddit.com/r/numerai/",
            "audience": "5,000+ NMR holders looking for automated tournament staking strategies",
            "intent_level": "MEDIUM",
            "status": "READY TO POST",
            "actionable_copy_file": "docs/NUMERAI_FORUM_PITCH.md",
            "strategy": "Post track record audit with CYPHER100 coupon code for retail NMR stakers",
        },
    ]
    assert isinstance(channels, list), "Channels must be a list"
    assert len(channels) == 4, "Must contain exactly 4 verified channels"
    return channels


def format_channel_action_guide() -> str:
    """Formats actionable markdown playbook for staker acquisition."""
    channels = get_channel_registry()
    assert isinstance(channels, list) and len(channels) > 0, "Channels must not be empty"
    lines = ["# High-Yield Target Audience Channels & Execution Playbook\n\n"]
    lines.append("| Priority | Channel / Community | Platform URL | Conversion Strategy | Action Asset |\n")
    lines.append("| :---: | :--- | :--- | :--- | :--- |\n")

    for idx, c in enumerate(channels):
        prio = f"**P{idx+1}**"
        lines.append(
            f"| {prio} | **{c['name']}** | [{c['platform']}]({c['url']}) | {c['strategy']} | `{c['actionable_copy_file']}` |\n"
        )

    lines.append("\n### Staker Timing Window\n")
    lines.append(
        "Optimal posting window is **Friday afternoon through Saturday 18:00 UTC** (before Round 1373 opens) "
        "when stakers actively scout models to lock into their upcoming weekend submissions with code **CYPHER100**.\n"
    )
    res = "".join(lines)
    assert len(res) > 100, "Output guide must not be empty"
    return res


def generate_channel_blueprint_file(output_path: str) -> str:
    """Generates the master TARGET_CHANNELS_BLUEPRINT.md file in docs directory."""
    assert isinstance(output_path, str) and len(output_path) > 0, "Invalid output_path"
    guide = format_channel_action_guide()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(guide)
    assert os.path.exists(output_path), "Target file must exist after write"
    return output_path


if __name__ == "__main__":
    out_file = os.path.join(BASE_DIR, "docs", "TARGET_CHANNELS_BLUEPRINT.md")
    saved = generate_channel_blueprint_file(out_file)
    print(f"Target channel blueprint generated at: {saved}")
