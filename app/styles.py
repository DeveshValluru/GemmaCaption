"""Style contracts, exemplars, prompt builders, and Tier-4 templates.

Single source of truth for the four registers (BLUEPRINT R-18). Mirrors
docs/03-style-playbook.md — keep them in sync. Exemplars are register references
only; runtime content always comes from the actual video (R-13).
"""
from __future__ import annotations

from .config import Config

# Official one-line definitions (verbatim from the guide, R-18).
STYLE_DEFS: dict[str, str] = {
    "formal": "Professional, objective, factual tone.",
    "sarcastic": "Dry, ironic, lightly mocking.",
    "humorous_tech": "Funny, with technology or programming references.",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon.",
}

# Do/don't contracts driving the stylizer.
STYLE_CONTRACTS: dict[str, str] = {
    "formal": (
        "Register: the precise, neutral, information-dense voice of a professional caption — a "
        "Reuters or AP news-wire caption, a Getty Images stock-footage description, or a natural-"
        "history documentary caption. State exactly what is shown, in order: the main subject with "
        "concrete attributes (colours, clothing, counts), its central action, the setting, and the "
        "most notable true details. Present tense, third person; no contractions, no opinion or "
        "emotion adjectives, no humour, irony, first person, or exclamation marks. Every clause is "
        "an independently checkable fact — write it so it could publish verbatim under a news photo."
    ),
    "sarcastic": (
        "Register: dry, deadpan, ironic — the withering wit of Oscar Wilde, Dorothy Parker, "
        "Groucho Marx, Winston Churchill, and Ricky Gervais. Pick ONE concrete thing actually "
        "visible in the scene and aim the whole caption at it with mock-admiration, faux-awe, or "
        "crushing understatement — the intensity of someone shutting down an obvious ragebaiter "
        "with a single unbothered line, or reacting to breathless hype about something utterly "
        "mundane. Confident, flat, devastating. The FACTS stay true (the subject and action are "
        "exactly what's on screen); only the framing is savage. Never invent drama, never random "
        "snark, never cruel about real people's looks or bodies; one flat killer line beats three. "
        "VARY the opening every time — do NOT start with 'Behold' (or 'Witness'/'Ah, yes'); open "
        "with a flat statement, a dry aside, or straight into the scene, so no two read from a template."
    ),
    "humorous_tech": (
        "Register: a comedy writer for a developer audience — the wit of XKCD, HBO's Silicon "
        "Valley, or dev Twitter. Land ONE genuinely funny, sustained technology/programming joke "
        "(deploys, debugging, merge conflicts, standups, latency, prod incidents, CI/CD, legacy "
        "code, rate limits) mapped PRECISELY onto what is literally happening in the clip, so the "
        "real scene stays obviously recognizable. The joke must actually land, not merely drop a "
        "buzzword; at most ~2 tech terms, no jargon salad. The facts stay true — the humour is the "
        "metaphor, never invented details. One sharp joke beats a pile of references. Map the "
        "metaphor onto VISIBLE actions and objects only — do NOT introduce off-screen items (unread "
        "messages, tickets, meetings, files) as if they were actually in the clip."
    ),
    "humorous_non_tech": (
        "Register: a warm observational comedian — the everyday-life wit of Jerry Seinfeld, Ellen "
        "DeGeneres, or a wholesome relatable stand-up ('what is the deal with...'). Find the funny "
        "in the mundane: anthropomorphize the subject, exaggerate tiny stakes, or frame a small "
        "human drama — all grounded in what is actually on screen. The punchline lands in the final "
        "clause. ABSOLUTE BAN on any technology, science, or internet word (app, AI, algorithm, "
        "wifi, update, download, battery, screen, code, robot, startup, notification, internet, "
        "digital, online, computer, phone). Everyday words only — food, moods, weather, chores, "
        "relationships. The facts stay true; the humour is in the framing, not invented details."
    ),
}

# Words that must never appear in humorous_non_tech (F-12-adjacent style guard).
NON_TECH_BANNED = [
    "app", "ai", "algorithm", "wifi", "wi-fi", "update", "download", "upload", "battery",
    "screen", "code", "coding", "robot", "startup", "notification", "internet", "digital",
    "online", "software", "hardware", "database", "server", "byte", "pixel", "wireless",
    "bluetooth", "smartphone", "laptop", "computer", "website", "cloud", "api",
]

EXEMPLARS: dict[str, list[str]] = {
    # DETAILED + grounded: multiple sentences that FULLY describe the scene (subjects, actions in
    # order, setting, specifics) written entirely in the style voice. Rich detail = high accuracy.
    "formal": [
        "An orange kitten emerges from dense garden foliage in dappled afternoon light, stepping over fallen leaves as it advances toward the camera. It pauses to sniff a low green stem before continuing along the flower bed, its movements deliberate against a softly blurred background of soil and stems.",
        "Steady traffic proceeds along a multi-lane, tree-lined city boulevard framed by yellowing autumn foliage and mid-rise buildings. Cars and a red bus move through the intersection at a moderate pace while pedestrians wait at the curb, the scene recorded from an elevated, stationary vantage point.",
    ],
    "sarcastic": [
        "Behold a truly harrowing wildlife expedition: one orange kitten bravely conquers roughly three feet of garden, pausing to interrogate a single leaf as though it owes him money, before resuming his perilous march across the flower bed with all the urgency of a creature who has never once been late for anything.",
        "Feast your eyes on this pulse-pounding thriller in which two hundred cars inch down a leafy boulevard at the blistering pace of continental drift, while a heroic red bus makes its grand entrance and the assembled pedestrians achieve the near-impossible feat of standing at a curb, waiting.",
    ],
    "humorous_tech": [
        "The orange kitten deploys straight from the garden bushes to the flower-bed production environment with zero staging and no rollback plan, pausing mid-rollout to run a quick sniff-test on a suspicious low leaf it has flagged as a critical bug, then advancing toward the camera like a long-running process that refuses to time out.",
        "City traffic attempts a massive distributed merge into the main intersection with no consensus algorithm and visibly high latency, two hundred vehicles and one ambitious red bus all pushing to the same branch at once, while the pedestrians at the curb wait on a permission prompt that never quite resolves.",
    ],
    "humorous_non_tech": [
        "An orange kitten sets off through the garden with the grim determination of a tiny general on a very important mission, stopping to inspect one unremarkable leaf like an inspector who has already decided the whole place is a disgrace, then marching across the flower bed as if he is personally coming to collect a debt.",
        "Two hundred drivers crawl down the leafy boulevard, each one alone in their little metal bubble and privately convinced that this lane, this one, is the genius move, while a red bus muscles into the intersection like it owns the whole street and everyone at the curb quietly rethinks their life choices.",
    ],
}

# Generic, content-free last-resort captions (Tier 4). Valid English, on-register,
# never claims specific facts we cannot see (R-13 safe — task-agnostic).
_TEMPLATES: dict[str, str] = {
    "formal": "A short video clip presents a real-world scene with people or objects in motion.",
    "sarcastic": "A riveting short clip, packed with all the drama a few passing seconds can muster.",
    "humorous_tech": "A brief clip that loads faster than a morning stand-up and resolves with fewer merge conflicts.",
    "humorous_non_tech": "A little slice of life, captured in the few seconds before anyone thought to tidy up.",
}


def style_def(style: str) -> str:
    return STYLE_DEFS.get(style, f"Caption written in a {style.replace('_', ' ')} tone.")


def style_contract(style: str) -> str:
    return STYLE_CONTRACTS.get(
        style,
        f"Register: {style.replace('_', ' ')}. Write a vivid caption in an unmistakable {style.replace('_', ' ')} tone.",
    )


def exemplars(style: str, k: int = 3) -> list[str]:
    return EXEMPLARS.get(style, [])[:k]


def template_caption(style: str, subject_hint: str = "") -> str:
    """Tier-4 fallback caption. Never empty, always English (R-10, R-12)."""
    base = _TEMPLATES.get(style)
    if base:
        return base
    return f"A short video clip, described here in a {style.replace('_', ' ')} tone."


def style_block(style: str, k_exemplars: int = 3) -> str:
    """Compact per-style instruction block reused across prompts.

    Exemplars are included only for styles in Config.EXEMPLAR_STYLES; omitting them for the
    witty styles removes a register anchor that may be limiting their variety (A/B lever).
    """
    banned = ""
    if style == "humorous_non_tech":
        banned = f"\nBANNED WORDS (never use any): {', '.join(NON_TECH_BANNED[:16])} ..."
    block = (
        f"STYLE: {style}\n"
        f"Definition: {style_def(style)}\n"
        f"Contract: {style_contract(style)}{banned}"
    )
    ex = exemplars(style, k_exemplars) if style in Config.EXEMPLAR_STYLES else []
    if ex:
        ex_txt = "\n".join(f"  - {e}" for e in ex)
        block += f"\nRegister exemplars (do NOT copy their content):\n{ex_txt}"
    return block
