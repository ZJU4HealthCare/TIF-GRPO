from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RewardConfig",
    "extract_section",
    "format_reward",
    "judge_match",
    "trajectory_integral_reward",
    "cal_llm_as_judge_score",
    "compute_score",
]


@dataclass
class RewardConfig:
    """All reward hyper-parameters and judge connection settings.

    Every coefficient of the TIF reward is exposed here so the scoring behaviour
    is fully reproducible and can be tuned without touching the code.
    """

    # ---- report structure ---------------------------------------------------
    # Section tags the model must emit, in order. Each ``s`` is scored against
    # ``extra_info["info"][f"{s}_abnormality_entity"]["abnormalities"]``.
    sections: Sequence[str] = ("findings", "impression")

    # ---- LLM judge (OpenAI-compatible endpoint) -----------------------------
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: float = 120.0
    max_retries: int = 3
    temperature: float = 0.0

    # ---- TIF reward coefficients (see trajectory_integral_reward) -----------
    alpha: float = 1.0          # weight of the running-cost (FN integral) term
    gamma: float = 1.0          # weight of the control-effort (FP penalty) term
    terminal_weight: float = 0.2  # weight of the terminal mean-alignment term
    explore_bonus: float = 0.05   # additive bonus when the model predicts anything
    partial_score: float = 0.5    # per-entity score when hit but only loc XOR attr
    eps: float = 1e-8

    # ---- section combination + length penalty -------------------------------
    format_weight: float = 0.1
    # Weight applied to each section score. If None, every section is weighted
    # equally so that ``sum(section_weights) == 1``.
    section_weight: Optional[float] = None
    length_penalty_threshold: int = 1600  # chars before the penalty kicks in
    length_penalty_scale: float = 5000.0  # larger -> gentler penalty
    fail_score: float = 0.0     # score for a section whose judge call fails

    @classmethod
    def from_env(cls, **overrides: Any) -> "RewardConfig":
        """Build a config from ``TIF_JUDGE_*`` env vars, with optional overrides."""
        cfg = cls(
            base_url=os.getenv("TIF_JUDGE_BASE_URL"),
            api_key=os.getenv("TIF_JUDGE_API_KEY", "EMPTY"),
            model=os.getenv("TIF_JUDGE_MODEL"),
            timeout=float(os.getenv("TIF_JUDGE_TIMEOUT", "120")),
            max_retries=int(os.getenv("TIF_JUDGE_MAX_RETRIES", "3")),
            temperature=float(os.getenv("TIF_JUDGE_TEMPERATURE", "0.0")),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def weight_for(self, section: str) -> float:
        if self.section_weight is not None:
            return self.section_weight
        return 1.0 / max(len(self.sections), 1)


_DEFAULT_CFG: Optional[RewardConfig] = None


def _default_cfg() -> RewardConfig:
    global _DEFAULT_CFG
    if _DEFAULT_CFG is None:
        _DEFAULT_CFG = RewardConfig.from_env()
    return _DEFAULT_CFG


def extract_section(text: str, tag: str) -> str:
    """Return the trimmed content of ``<tag>...</tag>`` ("" if absent)."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL)
    return match.group(1).strip() if match else ""


def format_reward(text: str, sections: Sequence[str]) -> float:
    """1.0 iff every section tag is present, non-overlapping and in order."""
    body = r".*?".join(rf"<{s}>.*?</{s}>" for s in sections)
    pattern = re.compile(rf"\s*{body}\s*", re.DOTALL)
    return 1.0 if pattern.fullmatch(text or "") else 0.0


def _extract_json(raw: str) -> Dict[str, Any]:
    """Parse the first JSON object out of a model response.

    Tolerates ```json fenced blocks and leading/trailing prose.
    """
    if raw is None:
        raise ValueError("empty judge response")
    cleaned = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object found in judge response: {raw[:200]!r}")
    return json.loads(cleaned[start : end + 1])


JUDGE_PROMPT = """You compare a model-predicted medical report section against a \
ground-truth abnormality set and output exactly one JSON object.

Inputs.
- Ground-truth abnormalities, already extracted as a structured JSON list: {gt}
- Model-predicted report section (free text): {pred}

Step 1 - extract abnormalities from the prediction.
- Use the provided ground-truth list directly as the reference abnormality set. \
Do not modify, merge, split, infer, or add any ground-truth entity.
- Extract abnormality entities from the predicted text under the same definition.
- Exclude explicitly negated findings, normal findings, examination-condition \
descriptions, and interpretive impressions not tied to an explicit imaging abnormality.

Step 2 - match. Match each ground-truth abnormality to at most one predicted \
abnormality based primarily on abnormality semantics. Medically equivalent or \
highly similar expressions may be matched, but no broadening, generalization, or \
inference beyond explicit textual evidence is allowed. Set hit=true if a matching \
abnormality is present, else hit=false. Any predicted abnormality that matches no \
ground-truth entity is a false positive.

Step 3 - consistency judgment.
- hit: true iff a medically equivalent abnormality is explicitly present in the prediction.
- location_match: true if the predicted anatomical location is medically consistent \
with the ground truth, or if neither side states a location. Must be false when hit=false.
- attribute_match: true if the predicted imaging/pathological attributes are medically \
consistent with the ground truth, or if neither side states attributes. Must be false when hit=false.
- Never infer missing locations or attributes.

Step 4 - aggregate. Report the result for every ground-truth abnormality and list \
every false positive.

Strict constraints. Do not hallucinate, assume, or strengthen uncertain expressions. \
Only explicit, text-supported abnormalities may be matched. Negated or ruled-out \
findings must not be extracted.

Output exactly one JSON object, no extra text:
{{
  "abnormalities": [
    {{"name": "<ground-truth abnormality name>", "hit": true, "location_match": true, "attribute_match": true}}
  ],
  "false_positive": [
    {{"name": "<false positive abnormality name>"}}
  ]
}}"""


def _judge_gt_payload(gt_abnormalities: Any) -> str:
    """Normalize the ground-truth abnormality list into a compact JSON string."""
    if isinstance(gt_abnormalities, str):
        return gt_abnormalities
    return json.dumps(gt_abnormalities, ensure_ascii=False)


def judge_match(pred_text: str, gt_abnormalities: Any, cfg: RewardConfig) -> Dict[str, Any]:
    """Call the LLM judge and return the structured comparison.

    Returns a dict of the form::

        {"abnormalities": [{"name", "hit", "location_match", "attribute_match"}, ...],
         "false_positive": [{"name"}, ...]}

    Raises on repeated failure so the caller can decide the fallback score.
    """
    from openai import OpenAI  # imported lazily so offline use needs no openai

    if not cfg.base_url or not cfg.model:
        raise RuntimeError(
            "LLM judge is not configured; set TIF_JUDGE_BASE_URL and TIF_JUDGE_MODEL"
        )

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY", timeout=cfg.timeout)
    prompt = JUDGE_PROMPT.format(gt=_judge_gt_payload(gt_abnormalities), pred=pred_text)

    last_err: Optional[Exception] = None
    for _ in range(max(cfg.max_retries, 1)):
        try:
            resp = client.chat.completions.create(
                model=cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
            )
            match = _extract_json(resp.choices[0].message.content)
            match.setdefault("abnormalities", [])
            match.setdefault("false_positive", [])
            return match
        except Exception as e:  # noqa: BLE001 - retry on any judge/parse failure
            last_err = e
    raise RuntimeError(f"judge failed after {cfg.max_retries} retries: {last_err}")


def _entity_alignment(item: Dict[str, Any], partial_score: float) -> float:
    """Per-entity alignment score r_i in {0, partial, 1}.

    - hit == False                       -> 0.0
    - hit and location & attribute match -> 1.0
    - hit and exactly one of them        -> partial_score
    - hit and neither                    -> 0.0
    """
    if not bool(item.get("hit", False)):
        return 0.0
    loc = bool(item.get("location_match", False))
    attr = bool(item.get("attribute_match", False))
    if loc and attr:
        return 1.0
    if loc or attr:
        return partial_score
    return 0.0


def trajectory_integral_reward(match: Dict[str, Any], cfg: Optional[RewardConfig] = None) -> float:
    """Collapse a structured GT/pred comparison into the scalar TIF reward.

    Given per-ground-truth-entity alignment scores ``r_1..r_K`` (K = number of GT
    abnormalities) and ``FP`` false positives, with ``M = #hit + FP`` the total
    predicted abnormality count, the reward is::

        R = alpha * (1 - (1/K) * sum_k (1 - (1/k) * sum_{i<=k} r_i)^2)   # FN integral
          + gamma * (1 - (FP / (M + eps))^2)                            # FP control effort
          + terminal_weight * (1/K) * sum_i r_i                         # terminal reward
          + explore_bonus * 1[M > 0]                                    # exploration bonus

    The first term integrates the running false-negative error
    ``E_fn_k = 1 - (1/k) sum_{i<=k} r_i`` along the abnormality trajectory, so a
    persistently missed abnormality is penalized at every step it stays missed.

    Edge case K == 0 (the report has no ground-truth abnormalities): there is no
    trajectory to integrate, so the reward reduces to rewarding a clean prediction
    and punishing any false positive: ``R = 2 - 2 * (FP / (FP + eps))^2``.
    """
    cfg = cfg or _default_cfg()
    abnormalities: List[Dict[str, Any]] = match.get("abnormalities") or []
    false_positive: List[Dict[str, Any]] = match.get("false_positive") or []

    fp = len(false_positive)
    hit_count = sum(1 for it in abnormalities if bool(it.get("hit", False)))
    m = hit_count + fp
    n = len(abnormalities)

    # --- edge case: no ground-truth abnormalities (normal report) ---
    if n == 0:
        u_fp = fp / (m + cfg.eps)
        return 2.0 - 2.0 * (u_fp ** 2)

    r_list = [_entity_alignment(it, cfg.partial_score) for it in abnormalities]

    # --- running cost: integral of the false-negative error over the trajectory ---
    cumsum = 0.0
    running_sq = 0.0
    for k, r_i in enumerate(r_list, start=1):
        cumsum += r_i
        e_fn = 1.0 - cumsum / k
        running_sq += e_fn ** 2
    running_cost = cfg.alpha * (1.0 - running_sq / n)

    # --- control effort: penalize false positives ---
    if m == 0:
        control_cost = 0.0
    else:
        u_fp = fp / (m + cfg.eps)
        control_cost = cfg.gamma * (1.0 - u_fp ** 2)

    # --- terminal reward + exploration bonus ---
    terminal = cfg.terminal_weight * (sum(r_list) / n)
    explore = cfg.explore_bonus if m > 0 else 0.0

    return running_cost + control_cost + terminal + explore


def cal_llm_as_judge_score(
    pred_text: str, gt_abnormalities: Any, cfg: Optional[RewardConfig] = None
) -> Dict[str, Any]:
    """Judge a single section and return its TIF score plus the raw comparison.

    Returns ``{"llm_as_judge_score": float, "match": dict}``.
    """
    cfg = cfg or _default_cfg()
    match = judge_match(pred_text, gt_abnormalities, cfg)
    return {"llm_as_judge_score": trajectory_integral_reward(match, cfg), "match": match}


def _gt_abnormalities(extra_info: Optional[Dict[str, Any]], section: str) -> Any:
    info = (extra_info or {}).get("info", {}) or {}
    entity = info.get(f"{section}_abnormality_entity", {}) or {}
    return entity.get("abnormalities", [])


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Dict[str, Any]] = None,
    cfg: Optional[RewardConfig] = None,
    **_: Any,
) -> float:
    """Reward for one generated report. Matches verl's dispatch signature.

    ``solution_str``  the model output, expected to contain the section tags.
    ``ground_truth``  the reference report string (kept for API compatibility;
                      structured scoring uses ``extra_info`` below).
    ``extra_info``    must carry the structured ground truth under
                      ``extra_info["info"][f"{section}_abnormality_entity"]["abnormalities"]``.
    """
    cfg = cfg or _default_cfg()

    fmt = format_reward(solution_str, cfg.sections)

    total_len = 0
    section_reward = 0.0
    for section in cfg.sections:
        pred_text = extract_section(solution_str, section)
        total_len += len(pred_text)
        gt = _gt_abnormalities(extra_info, section)
        try:
            score = cal_llm_as_judge_score(pred_text, gt, cfg)["llm_as_judge_score"]
        except Exception as e:  # noqa: BLE001 - a judge failure must not crash training
            print(f"[medical_report_abnormality] judge failed for <{section}>: {e}")
            score = cfg.fail_score
        section_reward += cfg.weight_for(section) * score

    raw = cfg.format_weight * fmt + section_reward

    overflow = max(total_len - cfg.length_penalty_threshold, 0)
    length_penalty = 1.0 / (1.0 + overflow / cfg.length_penalty_scale)
    return raw * length_penalty


if __name__ == "__main__":
    cfg = RewardConfig()

    # format_reward
    good = "<findings> a </findings>\n<impression> b </impression>"
    assert format_reward(good, cfg.sections) == 1.0
    assert format_reward("<findings> a </findings>", cfg.sections) == 0.0

    # trajectory_integral_reward on crafted comparisons
    all_hit = {
        "abnormalities": [
            {"name": "x", "hit": True, "location_match": True, "attribute_match": True},
            {"name": "y", "hit": True, "location_match": True, "attribute_match": True},
        ],
        "false_positive": [],
    }
    all_miss = {
        "abnormalities": [
            {"name": "x", "hit": False, "location_match": False, "attribute_match": False},
        ],
        "false_positive": [],
    }
    fp_only = {"abnormalities": [], "false_positive": [{"name": "z"}]}
    empty = {"abnormalities": [], "false_positive": []}

    r_all = trajectory_integral_reward(all_hit, cfg)
    r_miss = trajectory_integral_reward(all_miss, cfg)
    r_fp = trajectory_integral_reward(fp_only, cfg)
    r_empty = trajectory_integral_reward(empty, cfg)

    # all-hit: running=1, control=gamma=1, terminal=0.2, explore=0.05 -> 2.25
    assert abs(r_all - 2.25) < 1e-6, r_all
    # single miss: running=alpha*(1-1)=0, no prediction so control=0, terminal=0, explore=0
    assert abs(r_miss - 0.0) < 1e-6, r_miss
    # false positive only, no GT: 2 - 2*(1/(1+eps))^2 ~= 0
    assert r_fp < 1e-3, r_fp
    # empty vs empty (perfect normal report): 2.0
    assert abs(r_empty - 2.0) < 1e-6, r_empty

    print("all_hit =", r_all)
    print("all_miss =", r_miss)
    print("fp_only =", r_fp)
    print("empty =", r_empty)
    print("offline smoke test passed")
