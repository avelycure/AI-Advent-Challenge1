"""Входная и выходная политики — то, чего в проекте не было до агента."""
from __future__ import annotations

import pytest

from llmagent import InputPolicy, InputRejected, OutputPolicy, Schema
from llmagent.formats import Field
from llmagent.policies import apply_input, check_output, clean_output, output_stop

FIVE_BOOKS = Schema(fields=(Field("name", "строка", max_len=30),
                            Field("year", "целое число", minimum=1900, maximum=2030)),
                    item_count=2)


# --------------------------------------------------------------------------
# Вход
# --------------------------------------------------------------------------

def test_empty_request_never_reaches_provider():
    with pytest.raises(InputRejected):
        apply_input(InputPolicy(), "   \n  ")


def test_long_request_can_be_rejected():
    with pytest.raises(InputRejected) as info:
        apply_input(InputPolicy(max_chars=10), "x" * 11)
    assert "10" in str(info.value)


def test_long_request_can_be_trimmed_instead():
    assert apply_input(InputPolicy(max_chars=10, on_too_long="trim"), "x" * 20) == "x" * 10


def test_forbidden_substring_stops_the_request():
    policy = InputPolicy(forbidden=("пароль",))
    with pytest.raises(InputRejected):
        apply_input(policy, "Мой ПАРОЛЬ от почты")


def test_template_wraps_the_request():
    policy = InputPolicy(template="Переведи на английский: {input}")
    assert apply_input(policy, "  привет  ") == "Переведи на английский: привет"


# --------------------------------------------------------------------------
# Выход
# --------------------------------------------------------------------------

def test_free_format_asks_for_nothing_and_checks_nothing():
    policy = OutputPolicy()
    assert policy.instruction is None
    assert policy.response_format is None
    assert check_output(policy, "любой текст", "stop").ok


def test_json_policy_names_schema_in_the_instruction():
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS, stop_marker="%%END%%")
    instruction = policy.instruction
    assert "name" in instruction and "year" in instruction
    assert "Ровно 2 элементов" in instruction
    assert "%%END%%" in instruction
    assert policy.response_format == {"type": "json_object"}


def test_valid_json_passes_and_yields_data():
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS)
    body = '{"items": [{"name": "A", "year": 1990}, {"name": "B", "year": 2000}]}'
    result = check_output(policy, body, "stop")
    assert result.ok
    assert result.data["items"][1]["name"] == "B"


def test_wrong_item_count_is_caught():
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS)
    result = check_output(policy, '{"items": [{"name": "A", "year": 1990}]}', "stop")
    assert not result.ok
    assert "ровно 2" in result.failure_summary()


def test_wrong_type_is_caught():
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS)
    body = '{"items": [{"name": "A", "year": "1990"}, {"name": "B", "year": 2000}]}'
    assert not check_output(policy, body, "stop").ok


def test_value_out_of_range_is_caught():
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS)
    body = '{"items": [{"name": "A", "year": 1800}, {"name": "B", "year": 2000}]}'
    assert not check_output(policy, body, "stop").ok


def test_extra_field_is_caught_unless_allowed():
    strict = OutputPolicy(format="json", schema=FIVE_BOOKS)
    body = ('{"items": [{"name": "A", "year": 1990, "note": "x"}, '
            '{"name": "B", "year": 2000}]}')
    assert not check_output(strict, body, "stop").ok

    loose_schema = Schema(fields=FIVE_BOOKS.fields, item_count=2, allow_extra_fields=True)
    assert check_output(OutputPolicy(format="json", schema=loose_schema), body, "stop").ok


def test_fence_is_stripped_but_reported():
    """Живой GigaChat оборачивал JSON в тройные кавычки — данные при этом годны."""
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS)
    body = '```json\n{"items": [{"name": "A", "year": 1990}, {"name": "B", "year": 2000}]}\n```'
    result = check_output(policy, body, "stop")
    assert result.ok            # критичные проверки пройдены
    assert not result.flawless  # но замечание записано
    assert result.remarks


def test_marker_with_stray_newline_is_still_removed():
    """Модель разбивает маркер переносом — точное сравнение его не находит."""
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS, stop_marker="%%END%%")
    body = '{"items": [{"name": "A", "year": 1990}, {"name": "B", "year": 2000}]}\n%%EN\nD%%'
    assert check_output(policy, body, "stop").ok
    assert "%%" not in clean_output(policy, body)


def test_truncated_answer_is_caught_even_if_it_parses():
    policy = OutputPolicy(format="json", schema=FIVE_BOOKS)
    body = '{"items": [{"name": "A", "year": 1990}, {"name": "B", "year": 2000}]}'
    result = check_output(policy, body, "length")
    assert not result.ok


@pytest.mark.parametrize("fmt,body", [
    ("yaml", "items:\n  - name: A\n    year: 1990\n  - name: B\n    year: 2000"),
    ("md", "| name | year |\n|---|---|\n| A | 1990 |\n| B | 2000 |"),
])
def test_yaml_and_markdown_give_the_same_structure(fmt, body):
    policy = OutputPolicy(format=fmt, schema=FIVE_BOOKS)
    result = check_output(policy, body, "stop")
    assert result.ok
    assert result.data == {"items": [{"name": "A", "year": 1990},
                                     {"name": "B", "year": 2000}]}


def test_stop_marker_joins_user_stop_strings():
    policy = OutputPolicy(format="json", stop_marker="%%END%%")
    assert output_stop(policy, ["Вопрос:"]) == ["Вопрос:", "%%END%%"]
    assert output_stop(OutputPolicy(), None) is None
