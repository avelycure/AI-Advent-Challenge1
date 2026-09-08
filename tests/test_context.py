"""Сохранение контекста: имена сессий, поиск, возврат, краевые случаи."""
from __future__ import annotations

import json
import os
import pathlib
import time

import pytest

from llmagent import Agent, AgentConfig, SessionStore, Transport, restore_agent
from llmagent.store import SessionNotFound

from conftest import ScriptedClient

FAST_DEMO = Transport(demo=True, demo_delay=0.0)


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(str(tmp_path / "sessions"))


def talked(question: str, answer: str = "ответ") -> Agent:
    agent = Agent(AgentConfig(name="demo", transport=FAST_DEMO),
                  client=ScriptedClient(answer))
    agent.ask(question)
    return agent


# --------------------------------------------------------------------------
# Имя сессии
# --------------------------------------------------------------------------

def test_session_name_is_separate_from_the_agent_name():
    """Имя сессии говорит, о чём разговор; имя конфига — каким был агент."""
    from llmchat.session import Session

    session = Session(talked("вопрос"), title="Разбор алгоритмов")
    assert session.title == "Разбор алгоритмов"
    assert session.agent.config.name == "demo"


def test_a_name_is_suggested_from_the_topic():
    from llmchat.session import Session

    session = Session(talked("вопрос"), topic="Разбор алгоритмов")
    assert session.suggest_title() == "Разбор алгоритмов"


def test_a_name_is_suggested_from_the_first_question_when_there_is_no_topic():
    """Пока модель не придумала тему, начало вопроса — лучшее имя, что есть."""
    from llmchat.session import Session

    session = Session(talked("Как устроен алгоритм Дейкстры и когда он ломается?"))
    suggested = session.suggest_title()
    assert suggested.startswith("Как устроен алгоритм")
    assert len(suggested) <= 46


def test_a_note_is_not_taken_for_the_first_question():
    """Итог под-агента — не вопрос человека, и именем сессии быть не должен."""
    from llmchat.session import Session

    agent = Agent(AgentConfig(transport=FAST_DEMO), client=ScriptedClient("ответ"))
    agent.record_delegation("frugal", "чужой вопрос", "чужой ответ")
    agent.ask("мой настоящий вопрос")
    assert Session(agent).first_question == "мой настоящий вопрос"


def test_the_name_reaches_the_disk_and_comes_back(store):
    agent = talked("вопрос")
    store.save(agent, topic="тема", title="Моё имя")
    record = store.load(agent.session_id)
    assert record.title == "Моё имя"
    assert record.agent == "demo"
    assert record.label == "Моё имя"


def test_the_label_falls_back_in_order(store):
    """Имя, иначе тема, иначе первый вопрос — и никогда пустая строка."""
    agent = talked("Что такое хвостовая рекурсия?")
    store.save(agent, topic="", title="")
    assert store.load(agent.session_id).label == "Что такое хвостовая рекурсия?"

    store.save(agent, topic="Про рекурсию", title="")
    assert store.load(agent.session_id).label == "Про рекурсию"

    store.save(agent, topic="Про рекурсию", title="Имя")
    assert store.load(agent.session_id).label == "Имя"


# --------------------------------------------------------------------------
# Поиск и возврат
# --------------------------------------------------------------------------

def three_sessions(store) -> list:
    made = []
    for title, question in (("Алгоритмы", "Как устроен алгоритм Дейкстры?"),
                            ("Книги", "Посоветуй книги по алгоритмам"),
                            ("", "Что такое хвостовая рекурсия?")):
        agent = talked(question)
        store.save(agent, title=title)
        made.append(agent.session_id)
        # Пауза заметно больше точности времени: иначе порядок в списке
        # зависел бы от загрузки машины, а номера — от порядка.
        time.sleep(0.05)
    return made


def test_found_by_number_name_and_part_of_the_identifier(store):
    made = three_sessions(store)
    # Перечень кладём в сообщение: воспроизвести редкое падение не удалось,
    # и следующее должно объяснить себя само.
    listing = [(r.session_id, r.title, r.updated_at) for r in store.recent()]
    assert store.find("1").session_id == made[-1], (made, listing)
    assert store.find("Алгоритмы").session_id == made[0], (made, listing)
    # Префикс берём подлиннее: у четырёх шестнадцатеричных знаков есть шанс
    # совпасть у двух сессий, и тест стал бы неустойчивым.
    assert store.find(made[1][:6]).session_id == made[1]


def test_search_looks_inside_the_conversation(store):
    """Через неделю человек помнит, о чём говорил, а не как назвал сессию."""
    three_sessions(store)
    assert [r.label for r in store.matching("Дейкстры")] == ["Алгоритмы"]
    assert [r.label for r in store.matching("рекурсия")] == \
        ["Что такое хвостовая рекурсия?"]
    assert store.matching("такогослованет") == []


def test_search_without_a_word_returns_everything(store):
    three_sessions(store)
    assert len(store.matching("")) == 3


@pytest.mark.parametrize("wanted,expect", [
    ("9", "нет сессии под номером 9"),
    ("нетакого", "нет сессии «нетакого»"),
    ("", "не названа сессия"),
])
def test_mistakes_in_finding_are_reported(store, wanted, expect):
    three_sessions(store)
    with pytest.raises(SessionNotFound) as info:
        store.find(wanted)
    assert expect in str(info.value)


def test_an_ambiguous_name_asks_to_be_precise(store):
    for _ in range(2):
        agent = talked("вопрос")
        store.save(agent, title="Работа")
    with pytest.raises(SessionNotFound) as info:
        store.find("Работа")
    assert "подходит нескольким" in str(info.value)


def test_duplicate_names_stay_two_sessions(store):
    for question in ("первый", "второй"):
        agent = talked(question)
        store.save(agent, title="Работа")
    assert len(store.matching("Работа")) == 2


def test_removing_by_name_works_like_by_identifier(store):
    made = three_sessions(store)
    assert store.remove("Книги") is True
    assert len(store.recent()) == 2
    assert store.remove(made[0]) is True
    assert store.remove("Книги") is False


# --------------------------------------------------------------------------
# Краевые случаи
# --------------------------------------------------------------------------

def test_a_file_from_the_previous_version_still_reads(store):
    """Прежние файлы держали в «name» имя конфига, а поля «title» не знали."""
    agent = talked("Как устроен алгоритм Дейкстры?")
    store.save(agent, title="")
    path = store.path_for(agent.session_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    # У подлинно прежнего файла ни «title», ни «agent» не было вовсе.
    payload.pop("title")
    payload.pop("agent")
    payload["name"] = "старый-конфиг"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    record = store.load(agent.session_id)
    assert record.agent == "старый-конфиг"
    assert record.title == ""
    assert record.label == "Как устроен алгоритм Дейкстры?"


def test_a_broken_file_is_skipped_but_the_rest_are_read(store):
    agent = talked("живой разговор")
    store.save(agent, title="Живая")
    (store.root / "битая.json").write_text("{это не json", encoding="utf-8")
    assert [r.title for r in store.recent()] == ["Живая"]


def test_a_long_conversation_does_not_break_the_listing(store):
    agent = Agent(AgentConfig(transport=FAST_DEMO),
                  client=ScriptedClient(["ответ"] * 60))
    for number in range(60):
        agent.ask("вопрос номер {}".format(number))
    store.save(agent, title="Долгая")
    record = store.recent()[0]
    assert record.exchanges == 60
    assert record.first_question == "вопрос номер 0"


# --------------------------------------------------------------------------
# Две сессии в одном файле
# --------------------------------------------------------------------------

def test_a_second_terminal_does_not_clobber_the_first(store):
    """Открыть одну сессию дважды легко, и терять чужую работу нельзя."""
    first = talked("основа")
    store.save(first, title="Общая")
    seen = store.load(first.session_id).updated_at

    # Второй терминал успел записать свою правку.
    second = restore_agent(store.load(first.session_id), first.config,
                           client=ScriptedClient("ответ"))
    second.ask("из второго терминала")
    time.sleep(0.6)                      # у времени изменения грубая точность
    store.save(second, title="Общая")

    first.ask("из первого терминала")
    saved = store.save(first, title="Общая", seen_at=seen)

    assert saved.forked_from == second.session_id
    assert saved.session_id != second.session_id
    # Оба разговора целы, и ни один не потерян.
    kept = {r.session_id: [m["content"] for m in r.messages if m["role"] == "user"]
            for r in store.recent()}
    assert len(kept) == 2
    assert any("из второго терминала" in texts for texts in kept.values())
    assert any("из первого терминала" in texts for texts in kept.values())


def test_own_previous_save_is_not_taken_for_a_stranger(store):
    """Иначе сессия отделялась бы от себя самой после каждого ответа."""
    agent = talked("вопрос")
    saved = store.save(agent, title="Своя")
    for _ in range(3):
        agent.ask("ещё вопрос")
        saved = store.save(agent, title="Своя", seen_at=time.time())
        assert saved.forked_from == ""
    assert len(store.recent()) == 1


# --------------------------------------------------------------------------
# Найденное обзором кода. По тесту на каждую находку.
# --------------------------------------------------------------------------

def test_the_number_does_not_change_when_searching(store):
    """Номер — устойчивая ручка, иначе удаляли бы не ту сессию.

    Прежде список поиска нумеровался заново, а команды считали номер от
    полного списка: «--sessions Борщ» показывал «1», а «--rm-session 1»
    удалял совсем другой разговор.
    """
    made = three_sessions(store)
    full = {r.number: r.session_id for r in store.recent()}
    for record in store.matching("Дейкстры"):
        assert full[record.number] == record.session_id
        assert store.find(str(record.number)).session_id == record.session_id
    assert store.find(str(store.matching("Дейкстры")[0].number)).session_id == made[0]


def test_an_all_digit_identifier_still_opens(store):
    """Идентификатор из одних цифр бывает примерно у одной сессии из сорока."""
    agent = talked("вопрос")
    agent.session_id = "12345678"
    store.save(agent, title="Цифровая")
    assert store.find("12345678").title == "Цифровая"


def test_search_ignores_the_model_and_the_identifier(store):
    """Иначе по слову «gpt» находилось бы решительно всё."""
    three_sessions(store)
    assert store.matching("gpt") == []


def test_removing_an_ambiguous_name_is_an_error_not_a_guess(store):
    for _ in range(2):
        store.save(talked("вопрос"), title="Работа")
    with pytest.raises(SessionNotFound):
        store.remove("Работа")
    assert len(store.recent()) == 2


def test_a_concurrent_write_is_never_swallowed(store):
    """Допуск во времени был дырой: чужая запись в те же полсекунды пропадала."""
    first = talked("основа")
    saved = store.save(first, title="Общая")

    second = restore_agent(store.load(first.session_id), first.config,
                           client=ScriptedClient("ответ"))
    second.ask("из второго терминала")
    store.save(second, title="Общая")          # сразу, без всякой паузы

    first.ask("из первого терминала")
    again = store.save(first, title="Общая", seen_at=saved.updated_at)
    assert again.forked_from == first.session_id or again.session_id != saved.session_id
    assert len(store.recent()) == 2


def test_own_save_is_never_taken_for_a_stranger(store):
    """Время берётся из итога своей же записи, поэтому сессия не делится сама."""
    agent = talked("вопрос")
    saved = store.save(agent, title="Своя")
    for _ in range(4):
        agent.ask("ещё")
        saved = store.save(agent, title="Своя", seen_at=saved.updated_at)
        assert saved.forked_from == ""
    assert len(store.recent()) == 1


def test_concurrent_saves_do_not_share_a_temporary_file(store):
    """Общий временный файл сводил бы на нет всю затею с атомарной записью."""
    agent = talked("вопрос")
    store.save(agent, title="Своя")
    assert list(store.root.glob("*.tmp")) == []
    assert str(os.getpid()) in _temporary_name(store, agent.session_id)


def _temporary_name(store, session_id: str) -> str:
    return store.path_for(session_id).with_suffix(".{}.tmp".format(os.getpid())).name
