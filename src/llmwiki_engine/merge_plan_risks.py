from __future__ import annotations

from .models import WikiMergePlanArtifact, WikiMergePlanItem


def merge_plan_create_overlap_risk_items(plan: WikiMergePlanArtifact) -> list[WikiMergePlanItem]:
    return [
        item
        for item in plan.items
        if (item.model_action == "create" or item.action == "create")
        and item.strongest_overlap.strength in {"medium", "strong"}
        and item.strongest_overlap.path
    ]
