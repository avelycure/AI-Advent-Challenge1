"""Агент — коробка, в которую упакован весь путь от запроса до ответа.

Внутри по порядку: входная политика, сборка сообщений с историей, обращение
к провайдеру, выходная политика с переспросом, судья и учёт расхода. Снаружи
один метод ``ask`` и один конфиг.

Здесь нет ни печати, ни ввода: агент ничего не знает о том, кто его вызвал —
терминал, скрипт сравнения или HTTP-обработчик. Именно поэтому его можно
создать сотней штук из кода, не задав человеку ни одного вопроса.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from . import judge as judging
from . import policies
from .config import DEFAULT_CONFIG, AgentConfig
from .errors import OutputRejected
from .history import Conversation
from .registry import SHARED, ClientRegistry, Credentials, resolve_credentials
from .result import AgentResult
from .transport import Completion, count_message_tokens
from .usage import JUDGE, MAIN, REPAIR, SIDE, SUB, UsageMeter


class Agent:
    """Один агент — один конфиг, одна история, один счётчик расхода."""

    def __init__(self, config: AgentConfig = DEFAULT_CONFIG, *,
                 registry: Optional[ClientRegistry] = None, client: Any = None,
                 session_id: Optional[str] = None) -> None:
        # Проверяем конфиг сразу: неверный провайдер должен обнаружиться при
        # создании агента, а не на первом запросе где-то в глубине транспорта.
        config.resolve()
        self._config = config
        self._registry = registry if registry is not None else SHARED
        # Готовый клиент передают тесты и вызывающие, у которых он уже есть.
        self._client = client
        self._credentials: Optional[Credentials] = None
        # Своя сессия у каждого агента: память одного никак не пересекается
        # с памятью другого, и по идентификатору их видно на экране и в отчёте.
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.usage = UsageMeter()
        self.conversation = Conversation(keep_last_answer=config.history.keep_last_answer)
        self._judge_agent: Optional["Agent"] = None

    # --- конфиг ---------------------------------------------------------
    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    def reconfigure(self, **changes: Any) -> AgentConfig:
        """Заменить конфиг, сохранив историю и счётчики.

        Так меняются модель и параметры генерации прямо в работающем диалоге:
        конфиг заморожен, поэтому вместо правки поля создаётся новый.
        """
        updated = self._config.with_changes(**changes)
        updated.resolve()

        provider_changed = (updated.provider, updated.model) != (
            self._config.provider, self._config.model)
        credentials_changed = (updated.api_key, updated.api_extra, updated.transport) != (
            self._config.api_key, self._config.api_extra, self._config.transport)

        self._config = updated
        self.conversation.keep_last_answer = updated.history.keep_last_answer
        if provider_changed:
            # Точный размер контекста измерен токенизатором прежней модели
            # и после смены неверен: история снова оценивается локально.
            self.conversation.forget_exact()
        if provider_changed or credentials_changed:
            self._client = None
            self._credentials = None
        if "judge" in changes:
            self._judge_agent = None
        return updated

    # --- транспорт ------------------------------------------------------
    @property
    def credentials(self) -> Credentials:
        if self._credentials is None:
            self._credentials = resolve_credentials(self._config)
        return self._credentials

    @property
    def client(self):
        if self._client is None:
            self._client = self._registry.client_for(self._config, self.credentials)
        return self._client

    @property
    def model_ref(self) -> str:
        """Строка в поле ``model`` запроса: у Яндекса это длинный URI."""
        provider, model = self._config.resolve()
        return provider.model_ref(model, self.credentials.extra)

    def validate_credentials(self) -> None:
        """Дешёвая проверка ключа до начала работы; токенов не тратит."""
        self.client.validate_key()

    # --- окно контекста --------------------------------------------------
    @property
    def context_limit(self) -> int:
        return self._config.model_info.context_window

    @property
    def output_reserve(self) -> int:
        """Сколько токенов оставлено под ответ с учётом заданных параметров."""
        return self._config.generation.effective_max_tokens(
            self._config.model_info.output_reserve)

    @property
    def input_budget(self) -> int:
        """Сколько контекста доступно под историю с учётом места на ответ.

        Уменьшённый max_tokens освобождает место под историю — поэтому бюджет
        считается по действующему значению, а не по резерву модели.
        """
        return max(1, self.context_limit - self.output_reserve)

    def context_used(self) -> int:
        return self.conversation.context_used(self._config.system_prompt)

    def free_tokens(self) -> int:
        return max(0, self.input_budget - self.context_used())

    def fill_ratio(self) -> float:
        """Давление на бюджет истории: 1.0 — новые сообщения уже не примутся."""
        return min(1.0, self.context_used() / self.input_budget)

    def window_ratio(self) -> float:
        """Доля физического окна модели, занятая диалогом."""
        return min(1.0, self.context_used() / self.context_limit)

    def avg_exchange_tokens(self) -> int:
        """Средний прирост контекста за один обмен «вопрос — ответ»."""
        exchanges = self.conversation.exchanges
        if exchanges == 0:
            return 0
        return max(1, round(self.context_used() / exchanges))

    def remaining_exchanges(self) -> int:
        average = self.avg_exchange_tokens()
        return self.free_tokens() // average if average else 0

    def is_full(self) -> bool:
        return self.free_tokens() <= 0

    def reset(self) -> None:
        """Забыть диалог. Счётчики расхода остаются: деньги уже потрачены."""
        self.conversation.reset()

    def new_session(self) -> str:
        """Начать новую сессию: чистая память, новый счёт и новый идентификатор.

        Идентификатор меняется намеренно. Сохранённый разговор привязан к нему,
        и продолжи мы писать под тем же именем — прежняя переписка была бы
        затёрта первым же ответом нового диалога.

        Счётчики тоже обнуляются, в отличие от ``reset``. Расход прежнего
        разговора остался в его записи, и переносить его в новую сессию значило
        бы дважды показать одни и те же траты, а заодно съесть чужим расходом
        лимит из ``budget``.
        """
        self.conversation.reset()
        self.usage = UsageMeter()
        self.session_id = uuid.uuid4().hex[:8]
        return self.session_id

    # --- основной путь ----------------------------------------------------
    def prepare_input(self, text: str) -> str:
        """Пропустить запрос через входную политику, ничего не отправляя.

        Отдельным методом — ради интерфейса: он показывает вопрос в переписке
        до обращения к модели, и показывать надо именно то, что уйдёт.
        """
        return policies.apply_input(self._config.input, text)

    def ask(self, text: str) -> AgentResult:
        """Полный путь: политика входа → модель → политика выхода → судья.

        ``InputRejected`` и ``LLMError`` поднимаются наверх: вызывающий сам
        решает, показать их или превратить в результат с ошибкой. История при
        неудаче остаётся целой — неотвеченный вопрос из неё убирается.
        """
        prepared = self.prepare_input(text)
        if not self._config.history.enabled:
            return self._answer(self._plain_messages(prepared))

        self.conversation.add_user(prepared)
        try:
            return self.answer_pending()
        except Exception:
            self.conversation.drop_last_user()
            raise

    def answer_pending(self) -> AgentResult:
        """Ответить на вопрос, который уже лежит в истории.

        Вопрос из истории при неудаче не убирается: этот путь для вызывающего,
        который сам положил его туда и сам решает, что делать дальше.
        """
        return self._answer(self.conversation.api_messages(self._system_prompt()))

    def retry_last(self) -> Optional[AgentResult]:
        """Задать последний вопрос ещё раз — на той модели, что выбрана сейчас.

        В запрос уходит история по этот вопрос включительно: прежние ответы
        на него в неё не попадают, поэтому модели поставлена ровно та же задача.
        """
        index = self.conversation.last_user_index()
        if index < 0:
            return None
        messages = self.conversation.api_messages_upto(index, self._system_prompt())
        return self._answer(messages)

    def ask_messages(self, messages: List[Dict[str, str]], *, kind: str = MAIN,
                     max_tokens: Optional[int] = None,
                     temperature: Optional[float] = None,
                     stop: Optional[List[str]] = None,
                     response_format: Optional[Dict[str, str]] = None) -> Completion:
        """Один вызов модели готовыми сообщениями, мимо истории и политик.

        Нужен опытам, которые сами строят каждый запрос и сравнивают между
        собой постановки промпта. Учёт расхода и бюджет работают и здесь —
        иначе часть трат снова оказалась бы вне коробки.
        """
        return self._call(messages, kind=kind, max_tokens=max_tokens,
                          temperature=temperature, stop=stop,
                          response_format=response_format)

    # --- внутреннее -------------------------------------------------------
    def _system_prompt(self) -> str:
        return policies.output_system_prompt(self._config.output,
                                             self._config.system_prompt)

    def _plain_messages(self, text: str) -> List[Dict[str, str]]:
        prompt = self._system_prompt()
        messages = [{"role": "system", "content": prompt}] if prompt else []
        messages.append({"role": "user", "content": text})
        return messages

    def _call(self, messages: List[Dict[str, str]], kind: str,
              max_tokens: Optional[int] = None, temperature: Optional[float] = None,
              stop: Optional[List[str]] = None,
              response_format: Optional[Dict[str, str]] = None) -> Completion:
        """Обращение к провайдеру с проверкой бюджета и записью расхода."""
        params = self._config.generation
        reserve = max_tokens if max_tokens is not None else self.output_reserve

        self.usage.check(self._config.budget,
                         upcoming_tokens=count_message_tokens(messages) + reserve)

        provider, model = self._config.resolve()
        completion = self.client.complete(
            self.model_ref,
            messages,
            max_tokens=reserve,
            temperature=params.temperature if temperature is None else temperature,
            top_p=params.top_p,
            stop=policies.output_stop(self._config.output, params.stop) if stop is None else stop,
            response_format=(self._config.output.response_format
                             or params.response_format_arg) if response_format is None
            else response_format,
        )
        completion.cost = self.usage.record(completion, provider, model, kind)
        return completion

    def _answer(self, messages: List[Dict[str, str]]) -> AgentResult:
        """Спросить модель и, если форма не соблюдена, переспросить."""
        policy = self._config.output
        provider, model = self._config.resolve()
        transcript = list(messages)
        # Вопрос запоминаем до переспросов: после них последним сообщением
        # пользователя оказывается замечание о форме, а не сама задача.
        question = next((m["content"] for m in reversed(messages)
                         if m["role"] == "user"), "")

        attempt = 0
        completion = None
        validation = None
        dropped: List[str] = []
        notes: List[str] = []
        spent_seconds = 0.0
        prompt_tokens = completion_tokens = reasoning_tokens = 0
        cost: Optional[float] = None

        while attempt < policy.max_attempts:
            attempt += 1
            completion = self._call(transcript, kind=MAIN if attempt == 1 else REPAIR)
            spent_seconds += completion.elapsed
            prompt_tokens += completion.prompt_tokens
            completion_tokens += completion.completion_tokens
            reasoning_tokens += completion.reasoning_tokens
            cost = _add(cost, completion.cost)
            for name in completion.dropped_params:
                if name not in dropped:
                    dropped.append(name)
            for note in completion.notes:
                if note not in notes:
                    notes.append(note)

            validation = policies.check_output(policy, completion.text,
                                               completion.finish_reason)
            if validation.ok or attempt >= policy.max_attempts:
                break
            # Переспрос идёт в отдельной ветке разговора: негодный ответ и
            # замечание к нему в историю диалога не попадают.
            transcript = transcript + [
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": policies.repair_request(validation)},
            ]

        text = policies.clean_output(policy, completion.text)
        if policy.require_valid and not validation.ok:
            raise OutputRejected("ответ не прошёл проверку за {} {}: {}".format(
                attempt, "попытку" if attempt == 1 else "попытки",
                validation.failure_summary()))

        result = AgentResult(
            agent=self._config.name, provider=provider.name, model=model.id,
            text=text, data=validation.data, validation=validation, attempts=attempt,
            finish_reason=completion.finish_reason, dropped_params=dropped, notes=notes,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens, seconds=spent_seconds,
            cost=cost, currency=model.currency,
        )

        if self._config.history.enabled:
            message = self.conversation.add_assistant(text, model.id, provider.accent)
            message.elapsed = spent_seconds
            message.prompt_tokens = prompt_tokens
            message.completion_tokens = completion_tokens
            message.reasoning_tokens = reasoning_tokens
            message.cost = cost
            message.cost_currency = model.currency
            self.conversation.note_exchange(completion.prompt_tokens,
                                            completion.completion_tokens)

        if self._config.judge is not None:
            self._apply_judge(result, question)
        return result

    def _apply_judge(self, result: AgentResult, question: str) -> None:
        """Оценить ответ. Оценка — украшение: из-за неё запрос не пропадает.

        Ловим любую ошибку, а не только транспортную: к этому моменту ответ
        уже лежит в истории, и падение оценщика не должно её рвать.
        """
        config = self._config.judge
        prompt = judging.build_prompt(config, question, result.text)
        try:
            completion = self._judge_call(prompt, config.max_tokens)
        except Exception as exc:  # noqa: BLE001
            result.judge_note = str(exc)[:80]
            return
        result.scores = judging.parse_scores(config, completion.text)
        if not judging.complete_verdict(config, result.scores):
            result.judge_note = "оценщик ответил не по форме"
        elif judging.below_threshold(config, result.scores):
            result.judge_note = "средняя оценка ниже порога {}".format(config.min_mean)

    def _judge_call(self, prompt: str, max_tokens: int) -> Completion:
        """Спросить судью — своей моделью или отдельным агентом из его конфига."""
        messages = [{"role": "user", "content": prompt}]
        config = self._config.judge
        if config.agent is None:
            return self._call(messages, kind=JUDGE, max_tokens=max_tokens, temperature=0.0,
                              stop=[], response_format={})
        if self._judge_agent is None:
            # Судья судьи не бывает: вложенный конфиг обнуляется, иначе оценка
            # порождала бы оценку и так до упора.
            self._judge_agent = Agent(config.agent.with_changes(judge=None),
                                      registry=self._registry)
        completion = self._judge_agent.ask_messages(messages, kind=JUDGE,
                                                    max_tokens=max_tokens, temperature=0.0,
                                                    stop=[], response_format={})
        self.usage.absorb(self._judge_agent.usage)
        self._judge_agent.usage = UsageMeter()
        return completion

    def record_delegation(self, name: str, question: str, answer: str):
        """Принять в память сессии ответ под-агента.

        Под-агент — отдельная сессия со своей памятью, и его переписки мы не
        видим. В родительскую память попадает только итог, и попадает явно
        помеченным: дальше модель сможет на него ссылаться, но не спутает
        его со своим собственным ответом.
        """
        note = ("[под-агент {}] Ему был задан вопрос: {}\n\n"
                "Он ответил:\n{}".format(name, question, answer))
        return self.conversation.add_note(note, model=name)

    def absorb_delegation(self, prompt_tokens: int, completion_tokens: int,
                          seconds: float = 0.0, cost: Optional[float] = None,
                          currency: str = "USD") -> None:
        """Учесть расход под-агента в счётчике этой сессии."""
        self.usage.record_external(prompt_tokens, completion_tokens, seconds,
                                   cost, currency, kind=SUB)

    def record_side(self, completion: Completion) -> None:
        """Учесть расход на служебный запрос, сделанный вызывающим.

        Нужен интерфейсу: тему диалога он спрашивает сам, но платит за неё
        тот же кошелёк, и в общий счёт она обязана попасть.
        """
        provider, model = self._config.resolve()
        self.usage.record(completion, provider, model, SIDE)


def _add(current: Optional[float], addition: Optional[float]) -> Optional[float]:
    """Сложить стоимости, сохранив None как «цена неизвестна»."""
    if addition is None:
        return current
    return addition if current is None else current + addition
