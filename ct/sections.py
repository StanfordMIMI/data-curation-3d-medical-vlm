"""Organ-system sectioning of abdominal CT radiology reports.

Merlin-style findings sections are prefixed with an organ-system heading
("liver:", "kidneys and ureters:", ...). During training the text side of a
batch alternates between the full report and one extracted section, which
exposes the model to both global and organ-level supervision.
"""

import re
from typing import List, Tuple

# Heading variations, mapped to the canonical organ-system name. Order
# matters: it defines the regex alternation order used for matching.
_HEADING_VARIATIONS = {
    "lower thorax": "lower thorax",
    "lower chest": "lower thorax",
    "lung bases": "lower thorax",
    "liver": "liver",
    "liver and biliary tree": "liver",
    "biliary system": "liver",
    "gallbladder": "gallbladder",
    "spleen": "spleen",
    "pancreas": "pancreas",
    "adrenal glands": "adrenal glands",
    "adrenals": "adrenal glands",
    "kidneys": "kidneys",
    "kidneys and ureters": "kidneys",
    " gu ": "kidneys",
    "kidneys, ureters": "kidneys",
    "bowel": "bowel",
    "gastrointestinal tract": "bowel",
    " gi ": "bowel",
    "bowel/mesentery": "bowel",
    "peritoneal space": "peritoneum",
    "peritoneal cavity": "peritoneum",
    "abdominal wall": "peritoneum",
    "peritoneum": "peritoneum",
    "pelvic organs": "pelvic",
    "bladder": "pelvic",
    "prostate and seminal vesicles": "pelvic",
    "pelvis": "pelvic",
    "uterus and ovaries": "pelvic",
    "vasculature": "circulatory",
    "lymph nodes": "lymph nodes",
    "musculoskeletal": "musculoskeletal",
    "bones": "musculoskeletal",
}

# The organ systems sampled during training ("lymph nodes" is matched by the
# parser but not sampled as its own section).
ORGAN_SYSTEMS = [
    "lower thorax",
    "liver",
    "gallbladder",
    "spleen",
    "pancreas",
    "adrenal glands",
    "kidneys",
    "bowel",
    "peritoneum",
    "pelvic",
    "circulatory",
    "musculoskeletal",
]

_ALTERNATION = "|".join(_HEADING_VARIATIONS)
# The single-space alternative in the lookahead makes a section body end at
# ANY following `"word :"` heading, not only the known ones, so text under an
# unrecognized heading is never attributed to the preceding organ system.
_SECTION_PATTERN = re.compile(
    "(" + _ALTERNATION + ")\\s*:\\s*(.+?)(?=\\s*(?:" + _ALTERNATION + "| )\\s*:|$)",
    re.DOTALL,
)

# Length of the training-time section sampling cycle: even steps use the full
# report, odd steps walk through sections 1..11 of `ORGAN_SYSTEMS` in order.
SECTION_CYCLE_LENGTH = 23


def extract_section(findings: str, organ_system: str) -> str:
    """Extract one organ system's section from a findings report.

    Args:
        findings: The full findings text.
        organ_system: Canonical organ-system name from ``ORGAN_SYSTEMS``.

    Returns:
        The concatenated section text (heading included), or ``""`` when the
        report has no section for that organ system.
    """
    report = findings.lower()
    parts = []
    for heading, body in _SECTION_PATTERN.findall(report):
        if _HEADING_VARIATIONS[heading] == organ_system:
            parts.append(heading + ": " + body.strip())
    section = " ".join(parts)
    # The " gi "/" gu " heading variations match with a leading space.
    return section[1:] if section.startswith(" ") else section


def build_findings_list(findings: str) -> List[str]:
    """Build the per-sample text-candidate list used during training.

    Returns:
        ``[full_report, section_1, ..., section_12]`` following the order of
        ``ORGAN_SYSTEMS``; missing sections are empty strings.
    """
    return [findings] + [extract_section(findings, organ) for organ in ORGAN_SYSTEMS]


def section_index_for_step(step: int) -> int:
    """Index into a findings list for a given training step.

    The schedule alternates the full report (index 0) with one organ-system
    section, cycling with period ``SECTION_CYCLE_LENGTH``. This reproduces
    the paper's training recipe exactly.
    """
    position = step % SECTION_CYCLE_LENGTH
    if position % 2 == 0:
        return 0
    return (position + 1) // 2


def pick_training_texts(findings_lists: List[List[str]], step: int) -> List[str]:
    """Choose this step's text for each sample in a batch.

    Falls back to the full report when the sampled section is absent from a
    sample's report.

    Args:
        findings_lists: One findings list per sample (see
            ``build_findings_list``).
        step: Global training step within the epoch.

    Returns:
        One text string per sample.
    """
    index = section_index_for_step(step)
    return [texts[index] if texts[index] != "" else texts[0] for texts in findings_lists]
