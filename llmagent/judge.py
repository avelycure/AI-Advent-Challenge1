"""Оценка ответа отдельной моделью.

Судья ничего не вызывает сам: он собирает промпт и разбирает вердикт, а кто
именно спросит модель — дело агента. Так судья остаётся проверяемым без сети,
а агент волен судить и своей моделью, и чужой.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .config import JudgeConfig

JUDGE_PREAMBLE = (
    "Ты оцениваешь качество ответа, а не только его правильность.\n\n"
    "Вопрос: {question}\n\n"
    "Ответ, который надо оценить:\n---\n{answer}\n---\n"
)

# Оценщику даём начало и конец: по одной середине о полноте не судят.
HEAD_TAIL_LIMIT = 6000


def shorten(text: str, limit: int = HEAD_TAIL_LIMIT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n\n[…середина пропущена…]\n\n" + text[-half:]


def build_prompt(config: JudgeConfig, question: str, answer: str,
                 reference: Optional[str] = None) -> str:
    criteria = "\n".join("{} — {};".format(name, explain) for name, explain in config.criteria)
    shape = "\n".join("{}: <число>".format(name) for name, _ in config.criteria)
    parts = [JUDGE_PREAMBLE.format(question=question, answer=shorten(answer))]
    if reference:
        parts.append("Верный ответ: {}\n".format(reference))
    parts.append(
        "\nОцени по признакам целыми числами от 1 до {scale}:\n{criteria}\n\n"
        "Ответь строго {count} строками и ничем больше:\n{shape}".format(
            scale=config.scale, criteria=criteria, count=len(config.criteria), shape=shape))
    return "".join(parts)


def parse_scores(config: JudgeConfig, text: str) -> Dict[str, int]:
    scores: Dict[str, int] = {}
    for name, _ in config.criteria:
        match = re.search(re.escape(name) + r"\D{0,10}(\d+)", text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if 1 <= value <= config.scale:
                scores[name] = value
    return scores


def complete_verdict(config: JudgeConfig, scores: Dict[str, int]) -> bool:
    """Судья ответил по форме, только если назвал все признаки."""
    return len(scores) == len(config.criteria)


def mean(scores: Dict[str, int]) -> Optional[float]:
    return sum(scores.values()) / len(scores) if scores else None


def below_threshold(config: JudgeConfig, scores: Dict[str, int]) -> bool:
    average = mean(scores)
    return (config.min_mean is not None and average is not None
            and average < config.min_mean)


def criteria_names(config: JudgeConfig) -> List[str]:
    return [name for name, _ in config.criteria]
