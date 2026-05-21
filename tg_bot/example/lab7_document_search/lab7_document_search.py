# %% [markdown]
# # 🔍 Лабораторная работа №7: Поисковая система с ReAct агентом в LangGraph
# 
# ## 📚 Введение
# 
# Добро пожаловать в лабораторную работу, посвященную созданию интеллектуальной поисковой системы! 
# В этой работе мы объединим векторное хранилище LanceDB с технологией ReAct (Reasoning and Action) 
# в фреймворке LangGraph для создания агента, способного понимать запросы, рассуждать и находить 
# релевантную информацию в документах.
# 
# **Почему это важно?** Простой текстовый поиск не всегда способен понять контекст и намерения пользователя. 
# Интеллектуальные агенты на основе ReAct могут:
# * Анализировать запрос и выбирать оптимальную стратегию поиска
# * Комбинировать рассуждения с действиями для получения лучших результатов
# * Давать обоснованные ответы, опираясь на найденную информацию
# 
# В этой лабораторной работе мы будем использовать:
# * **LanceDB** - эффективное векторное хранилище для поиска документов
# * **LangGraph** - фреймворк для создания структурированных графов языковых моделей
# * **ReAct** - подход, сочетающий рассуждения (reasoning) и действия (action)
# 
# ## 🎯 Чему вы научитесь:
# 
# 1. **🧩 Работа с векторными базами данных** - как подключаться и выполнять поиск
# 2. **🔄 Создание ReAct агента** - реализация паттерна рассуждения и действия
# 3. **📊 Использование LangGraph** - построение графа состояний для управления агентом
# 4. **🚀 Инструменты для поиска** - разработка специализированных инструментов 
# 5. **💾 Сохранение состояния** - реализация персистентности между сессиями
# 
# Готовы создать учебного чат-бота по KOMPAS-3D? Давайте начнем!

# %% [markdown]
# ## Подготовка: Импорт библиотек

# %%
# Основные библиотеки
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv(".env")

# Импорт компонентов LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI

# Импорт компонентов для работы с векторным хранилищем
from langchain_community.vectorstores import LanceDB
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Импорт компонентов LangGraph
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict, List, Dict, Any, cast
import json
import os
import sys

# Инструменты для ReAct агента
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableConfig

_reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if _reconfigure_stdout:
    _reconfigure_stdout(encoding="utf-8")

CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "ollama").lower()
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:latest")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", CHAT_PROVIDER).lower()
MISTRAL_EMBEDDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")


def get_chat_model(temperature: float = 0.0):
    if CHAT_PROVIDER == "ollama":
        return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=temperature)
    return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=temperature)


def get_embeddings_model():
    if EMBEDDINGS_PROVIDER == "mistral":
        return MistralAIEmbeddings(model=MISTRAL_EMBEDDING_MODEL)
    if EMBEDDINGS_PROVIDER == "ollama":
        return OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    return OpenAIEmbeddings()

# %% [markdown]
# ## 1. Подключение к векторной базе данных
# 
# **💡 Почему это важно**: Векторная база данных хранит документы в виде математических векторов, что позволяет
# искать документы не только по точному соответствию ключевых слов, но и по семантической близости.
# 
# Мы будем использовать LanceDB с учебными материалами по моделированию в KOMPAS-3D.
# 
# **🔍 В этом разделе мы**:
# * Подключимся к существующей базе данных LanceDB
# * Проверим содержимое базы данных
# * Подготовим векторное хранилище для поиска

# %%
def connect_to_lancedb(db_path="../lb4/data/lancedb", table_name="pdf_docs"):
    """
    Подключение к базе знаний по обучению моделированию в KOMPAS-3D.
    
    Args:
        db_path: Путь к директории базы данных LanceDB
        table_name: Название таблицы в базе данных
    
    Returns:
        Экземпляр LanceDB для работы с векторным хранилищем
    """
    print(f"Подключение к базе данных LanceDB по пути: {db_path}")

    try:
        # Пытаемся импортировать lancedb
        import lancedb
        
        # Подключение к базе данных
        db = lancedb.connect(db_path)
        
        # Проверка существования таблицы
        table_names = db.list_tables().tables
        
        if not table_names:
            print(f"База данных не содержит таблиц. Путь: {db_path}")
            return None
        
        if table_name not in table_names:
            print(f"Таблица {table_name} не найдена. Доступные таблицы: {table_names}")
            return None
        
        # Открываем таблицу
        table = db.open_table(table_name)
        
        # Вывод информации о таблице
        print(f"Успешное подключение к таблице: {table_name}")
        print(f"Количество документов в базе: {table.count_rows()}")

        # Создаем модель эмбеддингов
        embeddings = get_embeddings_model()
        
        # Создаем экземпляр LanceDB для LangChain
        vector_store = LanceDB(
            connection=db,
            table_name=table_name,
            embedding=embeddings
        )
        
        return vector_store
    except ImportError:
        print("Ошибка импорта lancedb. Убедитесь, что библиотека установлена.")
        return None
    except Exception as e:
        print(f"Ошибка при подключении к базе данных: {str(e)}")
        return None

# %%
# Проверка подключения к базе данных
vector_db = connect_to_lancedb()

# %% [markdown]
# ## 2. Создание инструментов для поиска
# 
# **💡 Ключевой концепт**: Инструменты (tools) в ReAct позволяют агенту выполнять действия, такие как поиск информации.
# Хорошо продуманные инструменты делают агента более эффективным и целенаправленным.
# 
# **🔍 В этом разделе мы**:
# * Разработаем инструмент для стандартного векторного поиска
# * Создадим инструмент для поиска с фильтрацией по метаданным
# * Добавим инструмент для анализа найденных документов

# %%
# 
def raw_search_documents(query, vector_store, k=3):
    """
    Чистая функция поиска документов без зависимостей от инструментов LangChain.
    Эта функция предотвращает конфликты между объектом vector_store и системой обратных вызовов.
    
    Args:
        query (str): Текстовый запрос для поиска
        vector_store: Векторное хранилище LanceDB для поиска (LanceDB)
        k (int): Количество документов для возврата
    
    Returns:
        str: Отформатированная строка с найденными документами и их метаданными
    """
    try:
        
        
        # Выполнение поиска через LangChain API - семантический поиск по векторам
        results = vector_store.similarity_search(query, k=k)
        
        # Форматирование результатов в читаемый вид
        output = f'По запросу "{query}" найдено {len(results)} документов:\n\n'
        
        for i, doc in enumerate(results):
            output += f"Документ {i+1}:\n"
            output += f"{doc.page_content}\n\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                output += f"Метаданные: {doc.metadata}\n\n"
        
        return output
    except Exception as e:
        return f"Ошибка при выполнении поиска: {str(e)}"

def raw_search_with_filter(query, metadata_filter, vector_store, k=3):
    """
    Чистая функция поиска документов с применением фильтра по метаданным.
    Работает напрямую с векторным хранилищем без использования инструментов LangChain.
    
    Args:
        query (str): Текстовый запрос для поиска
        metadata_filter (dict): Словарь с фильтрами для метаданных
        vector_store: Векторное хранилище LanceDB для поиска
        k (int): Количество документов для возврата
    
    Returns:
        str: Отформатированная строка с найденными документами и их метаданными
    """
    try:
        results = vector_store.similarity_search(
            query=query, 
            k=k,
            filter=metadata_filter
        )
        
        # Форматирование результатов в читаемый вид
        output = f'По запросу "{query}" с фильтром {metadata_filter} найдено {len(results)} документов:\n\n'
        
        for i, doc in enumerate(results):
            output += f"Документ {i+1}:\n"
            output += f"{doc.page_content}\n\n"
            if hasattr(doc, 'metadata') and doc.metadata:
                output += f"Метаданные: {doc.metadata}\n\n"
        
        return output
    except Exception as e:
        return f"Произошла ошибка при поиске с фильтром: {str(e)}"

def raw_analyze_documents(documents):
    """
    Чистая функция анализа документов без связи с инструментами LangChain.
    Использует языковую модель для анализа содержимого документов.
    
    Args:
        documents (str): Строка с текстом документов для анализа
    
    Returns:
        str: Структурированный анализ документов с ключевой информацией
    """
    # Создаем шаблон запроса для анализа документов
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Вы - методист и эксперт по обучению 3D-моделированию в KOMPAS-3D.
        Проанализируйте найденные учебные материалы и выделите:
        1. Какие команды, панели или режимы KOMPAS-3D упоминаются
        2. Какую последовательность действий должен выполнить ученик
        3. Типичные ошибки и проверки результата
        4. Что стоит потренировать дополнительно
        
        Представьте анализ в формате короткой учебной подсказки."""),
        ("user", "{documents}")
    ])
    
    # Используем модель GPT-4o-mini для анализа документов
    model = get_chat_model(temperature=0)
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke({"documents": documents})

# %% [markdown]
# ## 3. Определение состояния ReAct агента
# 
# **💡 Ключевой концепт**: Состояние агента в LangGraph хранит всю необходимую информацию для принятия решений
# и выполнения действий. Для ReAct агента нам необходимо хранить историю сообщений, результаты поиска и
# информацию о текущем статусе обработки запроса.
# 
# **🔍 В этом разделе мы**:
# * Определим структуру состояния для учебного чат-бота
# * Создадим функцию для инициализации начального состояния

# %%
class SearchAgentState(TypedDict):
    """
    Состояние чат-бота-наставника по KOMPAS-3D на основе ReAct.
    """
    # Сообщения в диалоге (используем reducer add_messages для автоматического добавления)
    messages: Annotated[List[BaseMessage], add_messages]
    # Текущий статус обработки запроса
    status: str
    # Результаты последнего поиска
    search_results: List[str]
    # История поисковых запросов
    search_history: List[str]
    # Нормализованный запрос для векторного поиска
    search_query: str
    # Стратегия поиска: semantic или filtered
    search_strategy: str
    # Фильтр по метаданным LanceDB, если он применим
    metadata_filter: Dict[str, Any]
    # Определенная учебная тема KOMPAS-3D
    topic: str

def create_empty_state() -> SearchAgentState:
    """
    Создание начального пустого состояния для агента.
    """
    return {
        "messages": [],
        "status": "waiting_for_query",
        "search_results": [],
        "search_history": [],
        "search_query": "",
        "search_strategy": "semantic",
        "metadata_filter": {},
        "topic": "general"
    }

# %% [markdown]
# ## 4. Создание узлов для графа ReAct
# 
# **💡 Ключевой концепт**: ReAct состоит из рассуждений (reasoning) и действий (actions). В LangGraph
# это реализуется через отдельные узлы, которые выполняют различные функции в процессе обработки запроса.
# 
# **🔍 В этом разделе мы**:
# * Создадим узел для анализа запроса и планирования поиска
# * Реализуем узел для выполнения поиска
# * Добавим узел для формирования ответа на основе найденной информации





# %%  
def create_search_agent_nodes(vector_store):
    """
    Создание узлов для чат-бота-наставника по KOMPAS-3D на основе ReAct.
    
    Args:
        vector_store: Векторное хранилище для поиска
        
    Returns:
        Словарь с узлами графа
    """
    # Инициализация модели
    model = get_chat_model(temperature=0.2)
    
    # Вспомогательные функции для работы с чистыми функциями поиска
    def _search_func(query_str):
        """Прямая реализация поиска без вызова инструмента"""
        return raw_search_documents(query_str, vector_store)
    
    def _filter_search_func(args_str):
        """Прямая реализация поиска с фильтром без вызова инструмента"""
        try:
            args = json.loads(args_str)
            query = args.get("query", "")
            metadata_filter = args.get("metadata_filter", {})
            return raw_search_with_filter(query, metadata_filter, vector_store)
        except Exception as e:
            return f"Ошибка при обработке аргументов для фильтрованного поиска: {str(e)}"
    
    def _analyze_func(docs):
        """Прямая реализация анализа документов без вызова инструмента"""
        return raw_analyze_documents(docs)
    
    # Определяем инструменты, используя наши функции напрямую
    search_tool = Tool(
        name="search_documents",
        description="Поиск учебных материалов KOMPAS-3D по текстовому запросу",
        func=_search_func
    )
    
    filtered_search_tool = Tool(
        name="search_with_filter",
        description="Поиск учебных материалов KOMPAS-3D с фильтрацией по теме, уровню, типу операции или модулю",
        func=_filter_search_func
    )
    
    analyze_tool = Tool(
        name="analyze_documents",
        description="Анализ найденных материалов и превращение их в понятную учебную подсказку по KOMPAS-3D",
        func=_analyze_func
    )
    
    


    # Узел для выполнения поиска
    def execute_search(state: SearchAgentState) -> SearchAgentState:
        """Выполняет поиск на основе проанализированного запроса."""
        # Извлечение последнего запроса пользователя
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state  # Нет запросов пользователя
            
        query = str(user_messages[-1].content)

        if vector_store is None:
            return {
                "messages": state["messages"],
                "status": "knowledge_base_unavailable",
                "search_results": [],
                "search_history": state["search_history"] + [query],
                "search_query": state["search_query"] or query,
                "search_strategy": state["search_strategy"],
                "metadata_filter": state["metadata_filter"],
                "topic": state["topic"]
            }
        
        search_results: List[str] = []
        search_query = state["search_query"] or query
        metadata_filter = state["metadata_filter"]

        if state["search_strategy"] == "filtered" and metadata_filter:
            args_str = json.dumps({"query": search_query, "metadata_filter": metadata_filter})
            result = filtered_search_tool.invoke(args_str)
        else:
            result = search_tool.invoke(search_query)

        search_results.append(result)
        
        # Обновление истории поисков
        updated_history = state["search_history"] + [query]
        
        # Обновление состояния
        return {
            "messages": state["messages"],
            "status": "search_executed",
            "search_results": search_results,
            "search_history": updated_history,
            "search_query": search_query,
            "search_strategy": state["search_strategy"],
            "metadata_filter": metadata_filter,
            "topic": state["topic"]
        }
        
    # Импортируем узел анализа запроса и узел формирования ответа из предыдущего кода
    def analyze_query(state: SearchAgentState) -> SearchAgentState:
        """Анализирует запрос пользователя и определяет стратегию поиска."""
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state

        query = str(user_messages[-1].content)
        query_lower = query.lower()

        topic_keywords = {
            "sketch": ["эскиз", "контур", "размер", "зависим", "параметр"],
            "extrusion": ["выдав", "глубин", "элемент выдавливания", "операция выдавливания"],
            "revolution": ["вращ", "ось", "тело вращения"],
            "hole": ["отверст", "сверл", "резьб"],
            "fillet_chamfer": ["фаск", "скругл", "радиус"],
            "pattern": ["массив", "копи", "повтор"],
            "assembly": ["сборк", "сопряж", "компонент"],
            "drawing": ["чертеж", "вид", "разрез", "спецификац"],
            "import": ["импорт", "stp", "step", "iges", "x_t"],
        }

        topic_labels = {
            "sketch": "эскиз и параметризация",
            "extrusion": "операция выдавливания",
            "revolution": "операция вращения",
            "hole": "отверстия и резьба",
            "fillet_chamfer": "фаски и скругления",
            "pattern": "массивы и копирование",
            "assembly": "сборки и сопряжения",
            "drawing": "оформление чертежей",
            "import": "импорт геометрии",
        }

        topic = "general"
        for candidate, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topic = candidate
                break

        metadata_filter: Dict[str, Any] = {}
        strategy = "semantic"

        # В базе из lab4 доступны метаданные doc_name и chunk_id. Фильтр применяем
        # только когда пользователь явно указал имя документа через doc_name:...
        if "doc_name:" in query_lower:
            raw_doc_name = query.split("doc_name:", 1)[1].strip().split()[0]
            if raw_doc_name:
                metadata_filter = {"doc_name": raw_doc_name}
                strategy = "filtered"

        topic_hint = topic_labels.get(topic, "моделирование в KOMPAS-3D")
        search_query = f"{query} {topic_hint} KOMPAS-3D команды пошаговая инструкция"
        plan_text = (
            f"План поиска: тема - {topic_hint}; "
            f"стратегия - {'поиск с фильтром' if strategy == 'filtered' else 'семантический поиск'}; "
            f"поисковый запрос - {search_query}"
        )
        response = AIMessage(content=plan_text)

        return {
            "messages": state["messages"] + [response],
            "status": "query_analyzed",
            "search_results": state["search_results"],
            "search_history": state["search_history"],
            "search_query": search_query,
            "search_strategy": strategy,
            "metadata_filter": metadata_filter,
            "topic": topic
        }
    
    def generate_response(state: SearchAgentState) -> SearchAgentState:
        """Формирует ответ на основе результатов поиска."""
        # Если нет результатов поиска, возвращаем сообщение об этом
        if not state["search_results"]:
            if state["status"] == "knowledge_base_unavailable":
                response = AIMessage(content="База знаний по KOMPAS-3D сейчас не подключена, поэтому я не могу выполнить поиск по учебным материалам. Проверьте путь к LanceDB и таблицу с материалами, затем повторите вопрос.")
            else:
                response = AIMessage(content="Я не нашел подходящую подсказку по KOMPAS-3D в базе знаний. Уточните, пожалуйста, команду, тип модели или этап работы: эскиз, операция, сборка, чертеж или параметризация.")
            return {
                "messages": state["messages"] + [response],
                "status": "completed",
                "search_results": state["search_results"],
                "search_history": state["search_history"],
                "search_query": state["search_query"],
                "search_strategy": state["search_strategy"],
                "metadata_filter": state["metadata_filter"],
                "topic": state["topic"]
            }
        
        # Системное сообщение для формирования ответа
        system_message = SystemMessage(content="""
        Вы - чат-бот-наставник по обучению моделированию в KOMPAS-3D.
        Используйте результаты поиска, чтобы дать ученику практичный ответ по работе в KOMPAS-3D.
        
        Структурируйте свой ответ следующим образом:
        1. Коротко объясните, что нужно сделать
        2. Дайте пошаговую инструкцию в интерфейсе KOMPAS-3D
        3. Укажите важные параметры, ограничения и типичные ошибки
        4. Добавьте мини-задание для закрепления
        5. Укажите источники информации из найденных материалов
        
        Основывайтесь только на предоставленных результатах поиска.
        Если информации недостаточно, честно укажите, какой детали не хватает.
        """)
        
        # Создание контекста с результатами поиска
        search_context = "\n\n".join(state["search_results"])
        search_context_message = SystemMessage(content=f"Результаты поиска:\n{search_context}")
        
        # Получение последнего запроса пользователя
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        last_user_message = user_messages[-1] if user_messages else None
        
        # Если нет запроса, возвращаем состояние без изменений
        if not last_user_message:
            return state
        
        # Создание запроса для формирования ответа
        query_for_response = f"Вопрос ученика по KOMPAS-3D: {last_user_message.content}. Сформируйте учебный ответ на основе результатов поиска."
        
        # Сообщения для модели
        messages = [
            system_message,
            search_context_message,
            HumanMessage(content=query_for_response)
        ]
        
        # Вызов модели для формирования ответа
        response = model.invoke(messages)
        
        # Обновление состояния
        return {
            "messages": state["messages"] + [response],
            "status": "completed",
            "search_results": state["search_results"],
            "search_history": state["search_history"],
            "search_query": state["search_query"],
            "search_strategy": state["search_strategy"],
            "metadata_filter": state["metadata_filter"],
            "topic": state["topic"]
        }
    
    # Возвращаем словарь с узлами
    return {
        "analyze_query": analyze_query,
        "execute_search": execute_search,
        "generate_response": generate_response
    }

# %% [markdown]
# ## 5. Создание графа ReAct агента
# 
# **💡 Ключевой концепт**: Граф в LangGraph определяет, как узлы связаны между собой и какие переходы 
# возможны между ними. Для ReAct агента мы создадим граф с условными переходами, который позволит
# выполнять разные действия в зависимости от статуса обработки запроса.
# 
# **🔍 В этом разделе мы**:
# * Создадим граф состояний для агента
# * Определим условные переходы между узлами
# * Скомпилируем граф для дальнейшего использования

# %%
def create_search_agent_graph(vector_store):
    """
    Создание графа ReAct агента для поиска.
    
    Args:
        vector_store: Векторное хранилище для поиска
        
    Returns:
        Скомпилированный граф
    """
    # Создание графа с определенным типом состояния
    workflow = StateGraph(SearchAgentState)
    
    # Получение узлов
    nodes = create_search_agent_nodes(vector_store)
    
    # Добавление узлов в граф
    for name, function in nodes.items():
        workflow.add_node(name, function)
    
    # Определение условных переходов
    def should_search(state: SearchAgentState) -> str:
        """Определяет, нужно ли выполнять поиск или переходить к генерации ответа."""
        if state["status"] == "query_analyzed":
            return "execute_search"
        else:
            return "generate_response"
    
    # Добавление ребер с условными переходами
    workflow.add_edge(START, "analyze_query")
    workflow.add_conditional_edges("analyze_query", should_search)
    workflow.add_edge("execute_search", "generate_response")
    
    # Компиляция графа
    graph = workflow.compile()
    
    return graph

# %% [markdown]
# ## 6. Демонстрация работы ReAct агента
# 
# **💡 Ключевой концепт**: Теперь, когда мы создали все необходимые компоненты, пора увидеть нашего 
# агента в действии. Мы проверим, как он анализирует запросы, выполняет поиск и формирует ответы.
# 
# **🔍 В этом разделе мы**:
# * Проверим работу агента на различных типах запросов
# * Отследим прохождение запроса через все этапы обработки
# * Оценим качество ответов и их релевантность

# %%
def demonstrate_search_agent(vector_store=None):
    """
    Демонстрация работы чат-бота по KOMPAS-3D на основе ReAct.
    Показывает полный цикл обработки запроса: анализ, поиск и формирование ответа.
    
    Args:
        vector_store: Векторное хранилище для поиска
    
    Returns:
        dict: Конечное состояние агента после обработки всех запросов
    """
    # Создание графа для ReAct агента
    graph = create_search_agent_graph(vector_store)
    
    # Начальное состояние
    state = create_empty_state()
    
    # Список запросов для демонстрации различных сценариев поиска
    test_queries = [
        "Как построить деталь методом выдавливания из эскиза в KOMPAS-3D?",
        "Объясни, как сделать отверстие по центру цилиндрической детали",
        "Как создать сборку из двух деталей и задать сопряжения?"
    ]
    
    # Выполнение запросов
    for i, query in enumerate(test_queries):
        print(f"\n{'='*50}")
        print(f"ДЕМОНСТРАЦИЯ {i+1}: {query}")
        print(f"{'='*50}\n")
        
        # Добавление запроса пользователя в состояние
        state["messages"].append(HumanMessage(content=query))
        
        # Вызов графа для обработки запроса
        state = cast(SearchAgentState, graph.invoke(state))
        
        # Вывод ответа ассистента
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        last_message = ai_messages[-1] if ai_messages else None
        
        if last_message:
            print(f"📝 ОТВЕТ АГЕНТА:\n{last_message.content}")
        
        # Вывод информации о процессе обработки
        print(f"\n📊 СТАТИСТИКА:")
        print(f"• Статус: {state['status']}")
        print(f"• Найдено документов: {len(state['search_results'])}")
        print(f"• Выполнено поисков: {len(state['search_history'])}")
        print(f"{'='*50}\n")
    
    return state

# %% [markdown]
# ## 7. Сохранение состояния агента (персистентность)
# 
# **💡 Ключевой концепт**: Для создания полноценной поисковой системы нам нужно обеспечить сохранение
# состояния между сессиями пользователя. LangGraph поддерживает checkpointing, который позволяет
# сохранять и восстанавливать состояние агента.
# 
# **🔍 В этом разделе мы**:
# * Добавим механизм сохранения состояния с использованием MemorySaver
# * Продемонстрируем, как состояние можно восстановить между сессиями
# * Реализуем многопользовательский режим с использованием thread_id

# %%
def demonstrate_persistence():
    """
    Демонстрация персистентности с использованием чекпоинтеров в LangGraph.
    Показывает сохранение состояния между сессиями.
    """
    print("\n=== Демонстрация персистентности с чекпоинтерами ===\n")
    
    # Импортируем чекпоинтер для сохранения в памяти
    from langgraph.checkpoint.memory import MemorySaver
    
    # Определим простой процессор сообщений для демонстрации
    def simple_processor(state: SearchAgentState) -> SearchAgentState:
        """Простой процессор для демонстрации работы с состоянием."""
        # Получаем последнее сообщение пользователя
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return state
            
        last_message = str(user_messages[-1].content)
        
        # Добавляем ответ ассистента
        response = AIMessage(content=f"Я сохранил ваш вопрос по KOMPAS-3D: '{last_message}'. В следующем сообщении смогу учитывать этот контекст обучения.")
        
        # Обновляем историю поисковых запросов
        updated_history = state["search_history"] + [last_message]
        
        # Возвращаем обновленное состояние
        return {
            "messages": state["messages"] + [response],
            "status": "completed",
            "search_results": state["search_results"],
            "search_history": updated_history,
            "search_query": state["search_query"],
            "search_strategy": state["search_strategy"],
            "metadata_filter": state["metadata_filter"],
            "topic": state["topic"]
        }
    
    # Создание графа с определенным типом состояния
    workflow = StateGraph(SearchAgentState)
    
    # Добавляем узел в граф
    workflow.add_node("process_message", simple_processor)
    
    # Добавляем ребро от начала к узлу обработки
    workflow.add_edge(START, "process_message")
    
    # Создаем чекпоинтер в памяти
    memory_saver = MemorySaver()
    
    # Компилируем граф с чекпоинтером
    graph = workflow.compile(checkpointer=memory_saver)
    
    print("1️⃣ Создали граф с чекпоинтером MemorySaver")
    
    # Создаем начальное состояние
    state = create_empty_state()
    
    # Создаем уникальный идентификатор для сессии
    session_id = "demo_session_1"
    config: RunnableConfig = {"configurable": {"thread_id": session_id}}
    
    print("\n2️⃣ Первая сессия - отправляем сообщение")
    
    # Добавляем сообщение пользователя
    message1 = "Я учусь строить деталь в KOMPAS-3D и пока путаюсь между эскизом и операцией выдавливания"
    state["messages"].append(HumanMessage(content=message1))
    
    # Вызываем граф с конфигурацией для сохранения состояния
    print(f"👤 Пользователь: {message1}")
    state = cast(SearchAgentState, graph.invoke(state, config=config))
    
    # Выводим ответ
    ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    last_message = ai_messages[-1] if ai_messages else None
    
    if last_message:
        print(f"🤖 Ассистент: {last_message.content}")
    
    print(f"📊 История запросов: {state['search_history']}")
    
    print("\n3️⃣ Проверяем сохранение состояния")
    
    # Получаем сохраненное состояние из чекпоинтера
    state_snapshot = graph.get_state(config)
    
    if state_snapshot:
        print("✅ Состояние успешно сохранено!")
        print(f"📝 Количество сообщений в истории: {len(state_snapshot.values['messages'])}")
        print(f"🔍 История запросов: {state_snapshot.values['search_history']}")
    
    print("\n4️⃣ Имитируем новую сессию - восстанавливаем состояние и добавляем новое сообщение")
    
    # Получаем сохраненное состояние для новой сессии
    restored_state = cast(SearchAgentState, state_snapshot.values) if state_snapshot else create_empty_state()
    
    # Добавляем новое сообщение в восстановленное состояние
    message2 = "Теперь хочу понять, как после эскиза правильно задать глубину выдавливания"
    restored_state["messages"].append(HumanMessage(content=message2))
    
    # Вызываем граф с тем же thread_id
    print(f"👤 Пользователь: {message2}")
    updated_state = cast(SearchAgentState, graph.invoke(restored_state, config=config))
    
    # Выводим ответ
    ai_messages = [msg for msg in updated_state["messages"] if isinstance(msg, AIMessage)]
    last_message = ai_messages[-1] if ai_messages else None
    
    if last_message:
        print(f"🤖 Ассистент: {last_message.content}")
    
    print(f"📊 История запросов после восстановления: {updated_state['search_history']}")
    
    print("\n5️⃣ Итоговая проверка состояния")
    
    # Получаем финальное состояние из чекпоинтера
    final_snapshot = graph.get_state(config)
    
    if final_snapshot:
        print("✅ Итоговое состояние успешно сохранено!")
        print(f"📝 Количество сообщений в истории: {len(final_snapshot.values['messages'])}")
        
        # Выводим всю историю сообщений
        print("\n📜 Полная история диалога:")
        for i, msg in enumerate(final_snapshot.values["messages"]):
            if isinstance(msg, HumanMessage):
                print(f"   👤 Пользователь ({i+1}): {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"   🤖 Ассистент ({i+1}): {msg.content}")
    
    print("\n=== Демонстрация персистентности завершена ===")
    
    return memory_saver

# %% [markdown]
# ## 8. Поддержка множества пользователей
# 
# **💡 Ключевой концепт**: Для полноценной поисковой системы нужна поддержка нескольких пользователей
# одновременно. LangGraph позволяет это делать через использование разных thread_id для каждого пользователя.
# 
# **🔍 В этом разделе мы**:
# * Продемонстрируем, как обслуживать нескольких пользователей одновременно
# * Реализуем изоляцию состояний между пользователями

# %%
def demonstrate_multi_user_support(vector_store=None):
    """
    Демонстрация поддержки нескольких пользователей с изолированными состояниями.
    
    Args:
        vector_store: Векторное хранилище для поиска
    """
    from langgraph.checkpoint.memory import MemorySaver
    
    # Создание графа
    workflow = StateGraph(SearchAgentState)
    
    # Получение узлов
    nodes = create_search_agent_nodes(vector_store)
    
    # Добавление узлов
    for name, function in nodes.items():
        workflow.add_node(name, function)
    
    # Определение условных переходов
    def should_search(state: SearchAgentState) -> str:
        if state["status"] == "query_analyzed":
            return "execute_search"
        else:
            return "generate_response"
    
    # Добавление ребер
    workflow.add_edge(START, "analyze_query")
    workflow.add_conditional_edges("analyze_query", should_search)
    workflow.add_edge("execute_search", "generate_response")
    
    # Создание чекпоинтера
    memory_saver = MemorySaver()
    
    # Компиляция графа с чекпоинтером
    graph = workflow.compile(checkpointer=memory_saver)
    
    # Создание пользователей
    user_ids = ["user_1", "user_2"]
    user_states: Dict[str, SearchAgentState] = {}
    
    # Инициализация состояний для пользователей
    for user_id in user_ids:
        user_states[user_id] = create_empty_state()
    
    # Запросы для пользователей
    user_queries = {
        "user_1": [
            "Как создать параметрический эскиз прямоугольной пластины?",
            "Как потом изменить размеры пластины через переменные?"
        ],
        "user_2": [
            "Как добавить фаску на ребро детали?",
            "Как оформить чертеж по готовой 3D-модели?"
        ]
    }
    
    # Обработка запросов каждого пользователя
    for user_id in user_ids:
        print(f"\n=== Пользователь {user_id} ===\n")
        
        config: RunnableConfig = {"configurable": {"thread_id": user_id}}
        state = user_states[user_id]
        
        for query in user_queries[user_id]:
            # Добавление запроса в состояние
            state["messages"].append(HumanMessage(content=query))
            
            # Вызов графа с соответствующим thread_id
            state = cast(SearchAgentState, graph.invoke(state, config=config))
            
            # Вывод запроса и ответа
            print(f"Запрос: {query}")
            
            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            last_message = ai_messages[-1] if ai_messages else None
            
            if last_message:
                print(f"Ответ: {last_message.content}\n")
            
            # Обновление состояния пользователя
            user_states[user_id] = state
    
    # Проверка изоляции состояний
    print("\n=== Проверка изоляции состояний ===\n")
    
    for user_id in user_ids:
        # Получаем сохраненное состояние
        config: RunnableConfig = {"configurable": {"thread_id": user_id}}
        state_snapshot = graph.get_state(config)
        
        if state_snapshot:
            print(f"Пользователь {user_id}:")
            print(f"Количество сообщений: {len(state_snapshot.values['messages'])}")
            print(f"История поисковых запросов: {state_snapshot.values['search_history']}")
            print()
    
    return memory_saver

# %% [markdown]
# ## Заключение
# 
# **🎓 Что вы узнали в этой лаборатории**:
# 
# 1. **🧩 Интеграция с векторными базами данных** - как подключиться к LanceDB и выполнять поиск по документам
# 2. **🔄 Принципы паттерна ReAct** - как комбинировать рассуждения и действия для более эффективного поиска
# 3. **🚀 Создание агентов в LangGraph** - как определять состояние, узлы и переходы в графе
# 4. **💾 Сохранение состояния** - как реализовать персистентность и поддержку нескольких пользователей
# 

# %%
# Запуск демонстрации
if __name__ == "__main__":
    import lancedb
    import os
    
    # Путь к базе знаний с учебными материалами по KOMPAS-3D из лабораторной работы 4
    db_path = "../lb4/data/lancedb"
    
    # Проверяем, существует ли путь к БД
    if not os.path.exists(db_path):
        print(f"Путь к базе данных не существует: {db_path}")
        print("Используем тестовое окружение...")
        
        # Используем тестовое окружение
        vector_db = None
    else:
        try:
            # Подключаемся к базе данных
            vector_db = connect_to_lancedb(db_path=db_path, table_name="pdf_docs")
            if vector_db is not None:
                print("Успешное подключение к базе данных")
            else:
                print("База данных найдена, но таблица с учебными материалами недоступна.")
        except Exception as e:
            print(f"Ошибка при подключении к базе данных: {str(e)}")
            print("Используем тестовое окружение...")
            vector_db = None
    
    # Запускаем демонстрации
    print("\n=== Начинаем демонстрацию чат-бота по KOMPAS-3D ===\n")
    
    try:
        # Демонстрация чат-бота по KOMPAS-3D
        demonstrate_search_agent(vector_db)
    except Exception as e:
        print(f"Ошибка при демонстрации чат-бота по KOMPAS-3D: {str(e)}")
    
    try:
        # Демонстрация персистентности
        memory_saver = demonstrate_persistence()
        print("\nДемонстрация персистентности выполнена успешно.")
    except Exception as e:
        print(f"Ошибка при демонстрации персистентности: {str(e)}")

    try:
        # Демонстрация чат-бота для нескольких учеников
        demonstrate_multi_user_support(vector_db)
    except Exception as e:
        print(f"Ошибка при демонстрации чат-бота по KOMPAS-3D: {str(e)}")
    
    print("\n=== Демонстрации завершены ===\n")

# Узел для использования инструментов моделью
def create_tool_node(model_with_tools):
    """Создает узел графа для выполнения действий с помощью инструментов."""
    def tool_node(state: SearchAgentState) -> SearchAgentState:
        """Узел для работы с инструментами (tools)."""
        # Системное сообщение для определения использования инструментов
        system_message = SystemMessage(content="""
        Вы - чат-бот-наставник по обучению моделированию в KOMPAS-3D.
        Используйте инструменты для поиска учебных подсказок, пошаговых инструкций и объяснений команд.
        
        Исходя из предыдущего контекста и запроса пользователя, используйте инструменты:
        1. search_documents - для поиска материалов по вопросу ученика
        2. search_with_filter - для поиска с применением фильтров по теме, уровню или типу операции
        3. analyze_documents - для анализа найденных материалов и подготовки учебной подсказки
        
        Вызывайте инструменты, чтобы помочь ученику выполнить действие в KOMPAS-3D.
        """)
        
        # Подготовка сообщений (включая историю)
        messages = [system_message] + state["messages"]
        
        # Вызов модели с инструментами
        response = model_with_tools.invoke(messages)
        
        # Извлечение результатов использования инструментов
        search_results: List[str] = []
        
        # Проверяем вызовы инструментов в ответе
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                # Проверяем, является ли tool_call словарем (новый API) или объектом (старый API)
                if isinstance(tool_call, dict):
                    # Новый API - tool_call это словарь
                    if 'output' in tool_call:
                        search_results.append(tool_call['output'])
                else:
                    # Старый API - tool_call это объект с атрибутами
                    if hasattr(tool_call, "output"):
                        search_results.append(tool_call.output)
        
        # Обновление состояния
        return {
            "messages": state["messages"] + [response],
            "status": "tool_used" if search_results else "no_tool_used",
            "search_results": search_results if search_results else state["search_results"],
            "search_history": state["search_history"],
            "search_query": state["search_query"],
            "search_strategy": state["search_strategy"],
            "metadata_filter": state["metadata_filter"],
            "topic": state["topic"]
        }
    
    return tool_node
