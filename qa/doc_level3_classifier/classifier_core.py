# 文件作用：实现基于规则的文档三级标签分类核心逻辑。
# 关联说明：依赖 config、matching、profile，是规则分类的核心决策层。

from __future__ import annotations

from dataclasses import dataclass

from .config import DocClassifierConfig, LabelSpec, RuleSpec
from .matching import match_algorithm
from .profile import DocumentProfile


@dataclass(frozen=True)
class LabelScore:
    spec: LabelSpec
    score: float
    matched_rules: tuple[dict[str, object], ...]
    blocked_rules: tuple[dict[str, object], ...]


class DocumentRuleClassifier:
    def __init__(self, config: DocClassifierConfig) -> None:
        self.config = config

    def classify(self, profile: DocumentProfile) -> dict[str, object]:
        scored = [item for item in (self._score_label(profile, spec) for spec in self.config.labels) if item.score > 0]
        scored.sort(key=lambda item: (item.score, item.spec.priority), reverse=True)

        if not scored:
            return self._fallback_result(profile)

        level1_candidates = self._stage_candidates(scored, lambda item: item.spec.level1)
        selected_level1, level1_conf = self._pick_stage(level1_candidates)

        level2_source = [item for item in scored if item.spec.level1 == selected_level1]
        level2_candidates = self._stage_candidates(level2_source, lambda item: item.spec.level2)
        selected_level2, level2_conf = self._pick_stage(level2_candidates)

        level3_source = [item for item in level2_source if item.spec.level2 == selected_level2]
        level3_candidates = self._stage_candidates(level3_source, lambda item: item.spec.level3)
        selected_level3, level3_conf = self._pick_stage(level3_candidates)

        final = max(
            (item for item in level3_source if item.spec.level3 == selected_level3),
            key=lambda item: (item.score, item.spec.priority),
        )
        label_conf = self._confidence(final.score, scored[1].score if len(scored) > 1 else 0.0)

        return {
            "id": profile.source_id,
            "source_name": profile.source_name,
            "source_path": profile.source_path,
            "title": profile.title,
            "level1": final.spec.level1,
            "level2": final.spec.level2,
            "level3": final.spec.level3,
            "confidence": {
                "route": level1_conf,
                "level2": level2_conf,
                "level3": level3_conf,
                "label": label_conf,
            },
            "evidence": {
                "title": profile.title,
                "signals": list(profile.signals),
                "selected": {
                    "id": final.spec.id,
                    "score": round(final.score, 4),
                    "priority": final.spec.priority,
                    "matched_rules": list(final.matched_rules),
                    "blocked_rules": list(final.blocked_rules),
                },
                "top_candidates": [
                    {
                        "id": item.spec.id,
                        "level1": item.spec.level1,
                        "level2": item.spec.level2,
                        "level3": item.spec.level3,
                        "score": round(item.score, 4),
                    }
                    for item in scored[:5]
                ],
                "stage_candidates": {
                    "level1": level1_candidates[:5],
                    "level2": level2_candidates[:5],
                    "level3": level3_candidates[:5],
                },
            },
        }

    def _score_label(self, profile: DocumentProfile, spec: LabelSpec) -> LabelScore:
        score = 0.0
        matched_rules: list[dict[str, object]] = []
        blocked_rules: list[dict[str, object]] = []

        for rule in spec.rules:
            matched, detail = self._evaluate_rule(profile, rule)
            if matched:
                score += rule.weight
                matched_rules.append(
                    {
                        "scope": rule.scope,
                        "algorithm": rule.algorithm,
                        "weight": rule.weight,
                        "match": rule.match,
                        "detail": detail,
                    }
                )

        for rule in spec.excludes:
            matched, detail = self._evaluate_rule(profile, rule)
            if matched:
                score -= rule.weight
                blocked_rules.append(
                    {
                        "scope": rule.scope,
                        "algorithm": rule.algorithm,
                        "weight": rule.weight,
                        "match": rule.match,
                        "detail": detail,
                    }
                )

        score += spec.priority * 0.01
        if score < self.config.min_score:
            score = 0.0
        return LabelScore(spec=spec, score=score, matched_rules=tuple(matched_rules), blocked_rules=tuple(blocked_rules))

    def _evaluate_rule(self, profile: DocumentProfile, rule: RuleSpec) -> tuple[bool, dict[str, object]]:
        return match_algorithm(
            profile.scope_text(rule.scope),
            rule.algorithm,
            rule.match,
            case_sensitive=rule.case_sensitive,
            threshold=rule.threshold,
        )

    def _stage_candidates(self, scores: list[LabelScore], key_fn) -> list[dict[str, object]]:
        best_by_key: dict[str, LabelScore] = {}
        for item in scores:
            key = key_fn(item)
            current = best_by_key.get(key)
            if current is None or (item.score, item.spec.priority) > (current.score, current.spec.priority):
                best_by_key[key] = item
        ranked = sorted(best_by_key.items(), key=lambda pair: (pair[1].score, pair[1].spec.priority), reverse=True)
        return [{"name": name, "score": round(item.score, 4), "source_label": item.spec.id} for name, item in ranked]

    def _pick_stage(self, candidates: list[dict[str, object]]) -> tuple[str, float]:
        if not candidates:
            return "", 0.0
        top = float(candidates[0]["score"])
        second = float(candidates[1]["score"]) if len(candidates) > 1 else 0.0
        return str(candidates[0]["name"]), self._confidence(top, second)

    @staticmethod
    def _confidence(top: float, second: float) -> float:
        if top <= 0:
            return 0.0
        return round(top / (top + second + 1.0), 4)

    def _fallback_result(self, profile: DocumentProfile) -> dict[str, object]:
        return {
            "id": profile.source_id,
            "source_name": profile.source_name,
            "source_path": profile.source_path,
            "title": profile.title,
            "level1": self.config.fallback_label.level1,
            "level2": self.config.fallback_label.level2,
            "level3": self.config.fallback_label.level3,
            "confidence": {"route": 0.0, "level2": 0.0, "level3": 0.0, "label": 0.0},
            "evidence": {
                "title": profile.title,
                "signals": list(profile.signals),
                "selected": {
                    "id": "fallback",
                    "score": 0.0,
                    "priority": 0,
                    "matched_rules": [],
                    "blocked_rules": [],
                },
                "top_candidates": [],
                "stage_candidates": {"level1": [], "level2": [], "level3": []},
            },
        }
