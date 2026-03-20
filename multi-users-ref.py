"""
PDF 기반 멀티유저 멀티세션 RAG 챗봇
- Supabase Auth 기반 로그인/회원가입 (email=login id, password)
- 사용자별 세션 저장/로드
- Sidebar 상단에서 API 키 입력 (OpenAI, Anthropic, Gemini)
- Streamlit Cloud: SUPABASE_URL, SUPABASE_ANON_KEY는 Secrets에 등록 후 os.getenv로 읽음
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from supabase import create_client

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOGO_PATH = PROJECT_ROOT / "대전광역시.png"

BOT_NAME = "PDF 기반 멀티유저 멀티세션 RAG 챗봇"
MODEL_NAME = "gpt-4o-mini"

RAG_SEARCH_K = 10
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
PDF_BATCH_SIZE = 10

FOLLOW_UP_SECTION_TITLE = "### 💡 다음에 물어볼 수 있는 질문들"

ANSWER_STYLE_PROMPT = """
당신은 Supabase 기반 멀티유저 멀티세션 RAG 챗봇입니다.
답변 규칙을 반드시 지키세요.
1. 답변은 존대말의 서술형 문장으로 작성합니다.
2. 반드시 마크다운 헤딩 구조를 사용합니다.
3. 가장 큰 주제는 #, 세부 내용은 ##, 더 구체적인 설명은 ### 으로 구분합니다.
4. 구분선(---, ===, ___)을 사용하지 않습니다.
5. 취소선(~~텍스트~~)을 사용하지 않습니다.
6. 참조 표시, 출처 문구, 링크 목록, 각주를 작성하지 않습니다.
7. 문장을 끊어 쓰지 말고 의미가 완결되게 설명합니다.
""".strip()

LOGGER = logging.getLogger("multi_users_rag_chatbot")


def setup_logging() -> None:
    log_dir = LOG_DIR
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        # Streamlit Cloud 등 쓰기 권한이 없는 환경: /tmp 사용
        log_dir = Path("/tmp") / "multi_users_rag_logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            log_dir = None

    if not LOGGER.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)

        LOGGER.setLevel(logging.WARNING)
        LOGGER.addHandler(console_handler)

        if log_dir is not None:
            try:
                log_file = log_dir / f"multi_users_chatbot_{datetime.now().strftime('%Y%m%d')}.log"
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setLevel(logging.WARNING)
                file_handler.setFormatter(formatter)
                LOGGER.addHandler(file_handler)
            except OSError:
                pass

        LOGGER.propagate = False

    for noisy_logger in ("httpx", "httpcore", "urllib3", "openai", "langchain", "supabase"):
        logging.getLogger(noisy_logger).setLevel(logging.ERROR)
        logging.getLogger(noisy_logger).propagate = False


def apply_app_css() -> None:
    st.markdown(
        """
        <style>
        h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
        h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
        h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
        div[data-testid="stChatMessage"] {
            border-radius: 14px;
            padding: 0.4rem 0.8rem;
            background: rgba(255, 255, 255, 0.03);
        }
        div[data-testid="stChatMessageContent"] p,
        div[data-testid="stChatMessageContent"] li {
            line-height: 1.7;
        }
        button[kind="primary"],
        button[kind="secondary"] {
            background-color: #ff69b4 !important;
            color: white !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    left, center, right = st.columns([1.2, 4, 1])
    with left:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("<div style='font-size: 5rem;'>📚</div>", unsafe_allow_html=True)
    with center:
        st.markdown(
            """
            <div style="font-size: 3.2rem !important; font-weight: 800; margin-top: 0.6rem;">
                <span style="color: #1f77b4;">PDF 기반</span>
                <span style="color: #ff69b4;">멀티유저</span>
                <span style="color: #ffd700;">멀티세션 RAG 챗봇</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.write("")


def ensure_session_state() -> None:
    defaults: dict[str, Any] = {
        "supabase": None,
        "current_user": None,
        "active_session_id": None,
        "last_loaded_session_id": None,
        "chat_history": [],
        "processed_files": [],
        "openai_api_key": "",
        "anthropic_api_key": "",
        "gemini_api_key": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def remove_separators(text: str) -> str:
    cleaned = re.sub(r"~~(.*?)~~", r"\1", text, flags=re.DOTALL)
    cleaned = re.sub(r"(?m)^\s*([-_=])\1{2,}\s*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def create_supabase_client() -> Any:
    url = os.getenv("SUPABASE_URL")
    # 멀티유저 RLS 적용을 위해 ANON_KEY 사용 (SERVICE_ROLE_KEY는 RLS 우회)
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not anon_key:
        raise ValueError(
            "`SUPABASE_URL`, `SUPABASE_ANON_KEY`를 "
            "Streamlit Cloud Secrets 또는 환경변수에 설정해 주세요."
        )
    return create_client(url, anon_key)


def get_openai_api_key() -> str:
    key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY") or ""
    if not key:
        raise ValueError("OpenAI API 키를 사이드바에 입력하거나 환경변수에 설정해 주세요.")
    return key


def format_vector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    api_key = get_openai_api_key()
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
    )


def build_embeddings() -> OpenAIEmbeddings:
    api_key = get_openai_api_key()
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)


def create_session_row(supabase: Any, user_id: str, title: str | None) -> str:
    session_id = str(uuid.uuid4())
    payload = {
        "id": session_id,
        "user_id": user_id,
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
    }
    res = supabase.table("sessions").insert(payload).execute()
    if not res.data:
        raise RuntimeError("sessions row 생성 실패")
    return session_id


def fetch_sessions(supabase: Any) -> list[dict[str, Any]]:
    res = (
        supabase.table("sessions")
        .select("id,title,created_at")
        .order("created_at", desc=True)
        .limit(200)
        .execute()
    )
    return res.data or []


def load_session_chat_history(supabase: Any, session_id: str) -> list[dict[str, str]]:
    res = (
        supabase.table("chat_messages")
        .select("role,content,message_index")
        .eq("session_id", session_id)
        .order("message_index")
        .execute()
    )
    history: list[dict[str, str]] = []
    for row in res.data or []:
        history.append({"role": row["role"], "content": row["content"]})
    return history


def compute_next_message_index(supabase: Any, session_id: str) -> int:
    res = (
        supabase.table("chat_messages")
        .select("message_index")
        .eq("session_id", session_id)
        .order("message_index", desc=True)
        .limit(1)
        .execute()
    )
    data = res.data or []
    return 1 if not data else int(data[0]["message_index"]) + 1


def upsert_message(supabase: Any, session_id: str, role: str, content: str) -> None:
    next_idx = compute_next_message_index(supabase, session_id)
    supabase.table("chat_messages").insert(
        {
            "session_id": session_id,
            "message_index": next_idx,
            "role": role,
            "content": content,
        }
    ).execute()


def get_session_title(supabase: Any, session_id: str) -> str | None:
    res = supabase.table("sessions").select("title").eq("id", session_id).limit(1).execute()
    data = res.data or []
    if not data:
        return None
    title = data[0].get("title")
    return title if title else None


def update_session_title(supabase: Any, session_id: str, title: str) -> None:
    supabase.table("sessions").update({"title": title}).eq("id", session_id).execute()


def delete_session(supabase: Any, session_id: str) -> None:
    supabase.table("sessions").delete().eq("id", session_id).execute()


def clone_session_snapshot(
    supabase: Any, user_id: str, source_session_id: str, target_title: str
) -> None:
    target_session_id = str(uuid.uuid4())
    supabase.table("sessions").insert(
        {"id": target_session_id, "user_id": user_id, "title": target_title}
    ).execute()
    supabase.rpc(
        "clone_session_data",
        {"p_source_session_id": source_session_id, "p_target_session_id": target_session_id},
    ).execute()


def render_conversation_messages(history: list[dict[str, str]]) -> list[Any]:
    msgs: list[Any] = []
    for item in history:
        role = item.get("role")
        content = item.get("content") or ""
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs


def build_rag_context(rpc_rows: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for i, row in enumerate(rpc_rows, start=1):
        file_name = row.get("file_name") or "unknown"
        chunk_index = row.get("chunk_index")
        page_number = row.get("page_number")
        content = (row.get("content") or "").strip()
        if not content:
            continue

        meta_bits: list[str] = []
        meta_bits.append(f"조각 {chunk_index}" if chunk_index is not None else "조각")
        if page_number is not None:
            meta_bits.append(f"페이지 {page_number}")
        meta = " / ".join(meta_bits)
        chunks.append(f"[문서 조각 {i} / {file_name} / {meta}]\n{content}")
    return "\n\n".join(chunks)


def match_vector_documents(supabase: Any, session_id: str, query_text: str) -> list[dict[str, Any]]:
    embeddings = build_embeddings()
    q_vec = embeddings.embed_query(query_text)
    rpc_payload = {
        "p_session_id": session_id,
        "p_query_embedding": format_vector(q_vec),
        "p_match_count": RAG_SEARCH_K,
    }
    res = supabase.rpc("match_vector_documents", rpc_payload).execute()
    return res.data or []


def has_vector_documents(supabase: Any, session_id: str) -> bool:
    res = (
        supabase.table("vector_documents")
        .select("id")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def rag_answer_stream(
    supabase: Any,
    session_id: str,
    question: str,
    placeholder: Any,
    model: ChatOpenAI,
) -> str:
    rpc_rows = match_vector_documents(supabase, session_id, question)
    rag_context = build_rag_context(rpc_rows)

    system_prompt = ANSWER_STYLE_PROMPT + """

추가 규칙:
1. 아래에 제공된 문서 내용을 최우선으로 참고하여 답변합니다.
2. 문서에 없는 내용을 억지로 단정하지 않습니다.
3. 문서 맥락과 기존 대화 맥락을 함께 반영합니다.
""".strip()

    history_for_model = st.session_state.chat_history[:-1]

    user_prompt = question
    if rag_context:
        user_prompt = f"[문서 맥락]\n{rag_context}\n\n[사용자 질문]\n{question}"

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        *render_conversation_messages(history_for_model),
        HumanMessage(content=user_prompt),
    ]

    full_text = ""
    for chunk in model.stream(messages):
        full_text += str(getattr(chunk, "content", chunk) or "")
        placeholder.markdown(remove_separators(full_text))
    return remove_separators(full_text)


def direct_answer_stream(
    question: str,
    placeholder: Any,
    model: ChatOpenAI,
) -> str:
    system_prompt = ANSWER_STYLE_PROMPT
    history_for_model = st.session_state.chat_history[:-1]

    messages: list[Any] = [
        SystemMessage(content=system_prompt),
        *render_conversation_messages(history_for_model),
        HumanMessage(content=question),
    ]

    full_text = ""
    for chunk in model.stream(messages):
        full_text += str(getattr(chunk, "content", chunk) or "")
        placeholder.markdown(remove_separators(full_text))
    return remove_separators(full_text)


def generate_follow_up_questions(question: str, answer: str) -> list[str]:
    llm = get_llm(temperature=0.3)
    prompt = f"""
다음은 사용자의 질문과 답변입니다.

[질문]
{question}

[답변]
{answer}

사용자가 이어서 물어보면 좋은 질문 3개를 한국어로 생성하세요.
규칙:
1. 각 줄에는 질문만 하나씩 작성합니다.
2. 번호, 기호, 따옴표를 붙이지 않습니다.
3. 정확히 3줄만 출력합니다.
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content="당신은 후속 질문을 만드는 도우미입니다."),
            HumanMessage(content=prompt),
        ]
    )

    raw_lines = (getattr(response, "content", "") or "").splitlines()
    extracted: list[str] = []
    for line in raw_lines:
        cleaned = re.sub(r"^\s*[-\d\.\)]*\s*", "", line).strip()
        if not cleaned:
            continue
        if cleaned not in extracted:
            extracted.append(cleaned)
        if len(extracted) == 3:
            break

    fallback = [
        "이 내용과 관련된 핵심 정책이나 지원 제도를 알려주세요.",
        "실제로 신청하거나 이용하려면 어떤 절차를 따라야 하나요?",
        "주의해야 할 점이나 자주 묻는 질문도 함께 설명해 주세요.",
    ]
    return extracted if len(extracted) == 3 else fallback


def append_follow_up_section(answer: str, questions: list[str]) -> str:
    section_lines = [FOLLOW_UP_SECTION_TITLE]
    for index, question in enumerate(questions, start=1):
        section_lines.append(f"{index}. {question}")
    return remove_separators(f"{answer}\n\n" + "\n".join(section_lines))


def generate_session_title_from_first_qa(
    llm: ChatOpenAI, first_user_q: str, first_assistant_a: str
) -> str:
    prompt = f"""
아래는 대화의 첫 질문과 첫 답변입니다.
이 대화를 대표하는 세션 제목을 한국어로 만들어 주세요.

규칙:
1. 30자 이내로 짧고 명확하게
2. 반드시 질문/답변의 핵심 주제를 요약
3. 제목만 출력(추가 설명 금지)

[첫 질문]
{first_user_q}

[첫 답변]
{first_assistant_a}
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content="당신은 세션 제목 생성기입니다."),
            HumanMessage(content=prompt),
        ]
    )
    title = (getattr(response, "content", "") or "").strip().strip('"').strip("'")
    if len(title) > 60:
        title = title[:60].strip()
    return title or "저장된 세션"


def autogenerate_title_if_needed(supabase: Any, session_id: str) -> None:
    if get_session_title(supabase, session_id):
        return

    first_user = (
        supabase.table("chat_messages")
        .select("message_index,content")
        .eq("session_id", session_id)
        .eq("role", "user")
        .order("message_index")
        .limit(1)
        .execute()
    ).data
    if not first_user:
        return

    user_idx = int(first_user[0]["message_index"])
    first_assistant = (
        supabase.table("chat_messages")
        .select("message_index,content")
        .eq("session_id", session_id)
        .eq("role", "assistant")
        .gt("message_index", user_idx)
        .order("message_index")
        .limit(1)
        .execute()
    ).data
    if not first_assistant:
        return

    llm = get_llm(temperature=0.3)
    title = generate_session_title_from_first_qa(
        llm, first_user[0]["content"], first_assistant[0]["content"]
    )
    update_session_title(supabase, session_id, title)


def ensure_active_session_exists(supabase: Any, user_id: str) -> str:
    if st.session_state.active_session_id:
        return st.session_state.active_session_id
    session_id = create_session_row(supabase, user_id=user_id, title=None)
    st.session_state.active_session_id = session_id
    st.session_state.last_loaded_session_id = session_id
    st.session_state.chat_history = []
    st.session_state.processed_files = []
    return session_id


def handle_user_question(supabase: Any, user_id: str, question: str) -> None:
    session_id = ensure_active_session_exists(supabase, user_id)

    st.session_state.chat_history.append({"role": "user", "content": question})
    upsert_message(supabase, session_id, "user", question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            model = get_llm(temperature=0.5)
            if has_vector_documents(supabase, session_id):
                answer = rag_answer_stream(
                    supabase=supabase,
                    session_id=session_id,
                    question=question,
                    placeholder=placeholder,
                    model=model,
                )
            else:
                answer = direct_answer_stream(question=question, placeholder=placeholder, model=model)

            follow_ups = generate_follow_up_questions(question, answer)
            final_answer = append_follow_up_section(answer, follow_ups)
            placeholder.markdown(final_answer)

            st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
            upsert_message(supabase, session_id, "assistant", final_answer)

            autogenerate_title_if_needed(supabase, session_id)
        except Exception as exc:
            LOGGER.exception("질문 처리 실패: %s", exc)
            error_message = (
                "# 오류 안내\n\n질문을 처리하는 중 문제가 발생했습니다. "
                "API 키 설정과 상태를 다시 확인해 주세요."
            )
            placeholder.markdown(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
            upsert_message(supabase, session_id, "assistant", error_message)


def build_session_label(session: dict[str, Any]) -> str:
    title = session.get("title") or "제목 없음"
    created_at = session.get("created_at") or ""
    created_at_short = created_at.replace("T", " ").split(".")[0] if isinstance(created_at, str) else ""
    sid = str(session.get("id"))[:8]
    return f"{title} | {created_at_short} | {sid}"


def get_last_selected_session_id(selected_labels: list[str], sessions: list[dict[str, Any]]) -> str | None:
    if not selected_labels:
        return None
    label_to_id = {build_session_label(s): str(s["id"]) for s in sessions}
    return label_to_id.get(selected_labels[-1])


def render_chat_history() -> None:
    for message in st.session_state.chat_history:
        role = message.get("role")
        content = message.get("content") or ""
        if role not in ("user", "assistant"):
            continue
        with st.chat_message(role):
            st.markdown(content)


@contextmanager
def tempfile_for_uploaded_file(uploaded_file: Any) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getbuffer())
        yield temp_path


def file_sha256(uploaded_file: Any) -> str:
    buf = uploaded_file.getbuffer()
    h = hashlib.sha256()
    h.update(buf)
    return h.hexdigest()


def vector_already_exists_for_file(supabase: Any, session_id: str, file_sha: str) -> bool:
    res = (
        supabase.table("vector_documents")
        .select("id")
        .eq("session_id", session_id)
        .eq("file_sha256", file_sha)
        .limit(1)
        .execute()
    )
    return bool(res.data)


def insert_vector_documents(
    supabase: Any,
    session_id: str,
    uploaded_name: str,
    uploaded_file_sha: str,
    chunks: list[Any],
) -> None:
    embeddings = build_embeddings()

    to_insert_rows: list[dict[str, Any]] = []
    for chunk_idx, doc in enumerate(chunks):
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content:
            continue

        metadata = getattr(doc, "metadata", {}) or {}
        page_number = metadata.get("page") if isinstance(metadata, dict) else None

        to_insert_rows.append(
            {
                "session_id": session_id,
                "file_sha256": uploaded_file_sha,
                "file_name": uploaded_name,
                "chunk_index": chunk_idx,
                "page_number": page_number,
                "content": content,
            }
        )

    if not to_insert_rows:
        return

    for start in range(0, len(to_insert_rows), PDF_BATCH_SIZE):
        batch = to_insert_rows[start : start + PDF_BATCH_SIZE]
        texts = [row["content"] for row in batch]
        vecs = embeddings.embed_documents(texts)

        for row, vec in zip(batch, vecs):
            if len(vec) != EMBEDDING_DIM:
                raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(vec)}")
            row["embedding"] = format_vector(vec)

        supabase.table("vector_documents").insert(batch).execute()


def process_uploaded_pdfs(supabase: Any, session_id: str, uploaded_files: list[Any]) -> None:
    if not uploaded_files:
        st.warning("업로드된 PDF 파일이 없습니다.")
        return

    try:
        st.info("PDF를 처리하고 벡터를 저장하는 중입니다. 잠시만 기다려 주세요...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        processed_names: list[str] = []
        for uploaded_file in uploaded_files:
            uploaded_name = uploaded_file.name
            sha = file_sha256(uploaded_file)

            if vector_already_exists_for_file(supabase, session_id, sha):
                processed_names.append(uploaded_name)
                continue

            with tempfile_for_uploaded_file(uploaded_file) as temp_path:
                pages = PyPDFLoader(str(temp_path)).load()

            chunks = splitter.split_documents(pages)
            insert_vector_documents(
                supabase=supabase,
                session_id=session_id,
                uploaded_name=uploaded_name,
                uploaded_file_sha=sha,
                chunks=chunks,
            )
            processed_names.append(uploaded_name)

        st.success(f"{len(processed_names)}개 PDF의 벡터 처리를 완료했습니다.")
        st.session_state.processed_files = processed_names
    except Exception as exc:
        LOGGER.exception("PDF 처리 실패: %s", exc)
        st.error("PDF 처리 중 오류가 발생했습니다. 파일 형식과 API 키를 확인해 주세요.")


def render_api_keys_section() -> None:
    st.sidebar.markdown("### 🔑 API 키 설정")
    st.sidebar.caption("멀티유저 환경: .env 대신 여기서 입력 (또는 Streamlit Secrets)")

    openai_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=st.session_state.openai_api_key,
        type="password",
        key="sidebar_openai_key",
        placeholder="sk-...",
    )
    if openai_key:
        st.session_state.openai_api_key = openai_key

    anthropic_key = st.sidebar.text_input(
        "Anthropic API Key (선택)",
        value=st.session_state.anthropic_api_key,
        type="password",
        key="sidebar_anthropic_key",
        placeholder="sk-ant-...",
    )
    if anthropic_key:
        st.session_state.anthropic_api_key = anthropic_key

    gemini_key = st.sidebar.text_input(
        "Gemini API Key (선택)",
        value=st.session_state.gemini_api_key,
        type="password",
        key="sidebar_gemini_key",
        placeholder="AIza...",
    )
    if gemini_key:
        st.session_state.gemini_api_key = gemini_key

    st.sidebar.markdown("---")


def render_auth_section(supabase: Any) -> bool:
    """로그인/회원가입 UI. 로그인 성공 시 True 반환."""
    st.sidebar.markdown("### 👤 로그인 / 회원가입")
    st.sidebar.caption("Supabase Auth: 이메일(로그인 ID)과 비밀번호 사용")

    auth_tab = st.sidebar.radio("", ["로그인", "회원가입"], key="auth_tab", horizontal=True)

    login_id = st.sidebar.text_input("이메일 (로그인 ID)", key="auth_login_id", placeholder="user@example.com")
    password = st.sidebar.text_input("비밀번호", type="password", key="auth_password")

    if auth_tab == "로그인":
        if st.sidebar.button("로그인", use_container_width=True):
            if not login_id or not password:
                st.sidebar.error("이메일과 비밀번호를 입력해 주세요.")
            else:
                try:
                    res = supabase.auth.sign_in_with_password({"email": login_id, "password": password})
                    if res.user and res.session:
                        st.session_state.current_user = res.user
                        st.sidebar.success(f"로그인 성공: {res.user.email}")
                        st.rerun()
                    else:
                        st.sidebar.error("로그인에 실패했습니다.")
                except Exception as e:
                    st.sidebar.error(f"로그인 오류: {str(e)}")

    else:
        if st.sidebar.button("회원가입", use_container_width=True):
            if not login_id or not password:
                st.sidebar.error("이메일과 비밀번호를 입력해 주세요.")
            elif len(password) < 6:
                st.sidebar.error("비밀번호는 6자 이상이어야 합니다.")
            else:
                try:
                    res = supabase.auth.sign_up({"email": login_id, "password": password})
                    if res.user:
                        st.sidebar.success(
                            f"회원가입 완료: {res.user.email}. "
                            "이메일 확인이 필요할 수 있습니다."
                        )
                        if res.session:
                            st.session_state.current_user = res.user
                            st.rerun()
                    else:
                        st.sidebar.error("회원가입에 실패했습니다.")
                except Exception as e:
                    st.sidebar.error(f"회원가입 오류: {str(e)}")

    return False


def render_logged_in_sidebar(supabase: Any, user: Any) -> None:
    st.sidebar.success(f"로그인: {user.email}")
    if st.sidebar.button("로그아웃", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state.current_user = None
        st.session_state.active_session_id = None
        st.session_state.last_loaded_session_id = None
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.rerun()

    st.sidebar.markdown("---")


def render_sidebar(supabase: Any, user_id: str) -> None:
    sessions = fetch_sessions(supabase)
    session_labels = [build_session_label(s) for s in sessions]
    active_id = st.session_state.active_session_id
    active_label = None
    if active_id:
        for s in sessions:
            if str(s["id"]) == active_id:
                active_label = build_session_label(s)
                break

    selected_labels = st.sidebar.multiselect(
        "세션 선택",
        options=session_labels,
        default=[active_label] if active_label else [],
        key="session_multiselect",
        help="세션을 선택하면 자동으로 해당 세션을 로드합니다.",
    )

    selected_session_id = get_last_selected_session_id(selected_labels, sessions)
    if selected_session_id and selected_session_id != st.session_state.last_loaded_session_id:
        st.session_state.active_session_id = selected_session_id
        st.session_state.last_loaded_session_id = selected_session_id
        st.session_state.chat_history = load_session_chat_history(supabase, selected_session_id)
        st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("세션저장", use_container_width=True):
        session_id = ensure_active_session_exists(supabase, user_id)

        first_user = (
            supabase.table("chat_messages")
            .select("content,message_index")
            .eq("session_id", session_id)
            .eq("role", "user")
            .order("message_index")
            .limit(1)
            .execute()
        ).data
        first_assistant = (
            supabase.table("chat_messages")
            .select("content,message_index")
            .eq("session_id", session_id)
            .eq("role", "assistant")
            .order("message_index")
            .limit(1)
            .execute()
        ).data

        if not first_user or not first_assistant:
            st.warning("첫 질문/답변이 아직 저장되지 않았습니다. 먼저 대화를 시작해 주세요.")
        else:
            llm = get_llm(temperature=0.3)
            title = generate_session_title_from_first_qa(
                llm, first_user[0]["content"], first_assistant[0]["content"]
            )
            clone_session_snapshot(supabase, user_id, session_id, title)
            st.success("세션을 저장했습니다.")
            st.rerun()

    if st.sidebar.button("세션로드", use_container_width=True):
        if not selected_session_id:
            st.warning("세션을 먼저 선택해 주세요.")
        else:
            st.session_state.active_session_id = selected_session_id
            st.session_state.last_loaded_session_id = selected_session_id
            st.session_state.chat_history = load_session_chat_history(supabase, selected_session_id)
            st.rerun()

    if st.sidebar.button("세션삭제", use_container_width=True):
        if not selected_session_id:
            st.warning("세션을 먼저 선택해 주세요.")
        else:
            delete_session(supabase, selected_session_id)
            st.session_state.active_session_id = None
            st.session_state.last_loaded_session_id = None
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.rerun()

    if st.sidebar.button("화면초기화", use_container_width=True):
        st.session_state.active_session_id = None
        st.session_state.last_loaded_session_id = None
        st.session_state.chat_history = []
        st.session_state.processed_files = []
        st.rerun()

    st.sidebar.markdown("---")

    uploaded_files = st.sidebar.file_uploader(
        "PDF 업로드 (세션 벡터 저장)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.sidebar.button("파일 처리하기", use_container_width=True):
        session_id = ensure_active_session_exists(supabase, user_id)
        process_uploaded_pdfs(supabase, session_id, uploaded_files or [])

    if st.session_state.processed_files:
        st.sidebar.markdown("**처리된 파일 목록**")
        st.sidebar.text("\n".join(st.session_state.processed_files))

    st.sidebar.markdown("---")

    if st.sidebar.button("vectordb", use_container_width=True):
        session_id = st.session_state.active_session_id
        if not session_id:
            st.warning("먼저 세션을 활성화(대화 시작 또는 파일 처리)해 주세요.")
        else:
            res = (
                supabase.table("vector_documents")
                .select("file_name")
                .eq("session_id", session_id)
                .execute()
            )
            file_names = sorted({row.get("file_name") for row in (res.data or []) if row.get("file_name")})
            st.sidebar.markdown("**현재 vectordb 파일명**")
            st.sidebar.text("\n".join(file_names) if file_names else "저장된 벡터 문서가 없습니다.")

    st.sidebar.markdown("---")
    active_id = st.session_state.active_session_id
    st.sidebar.caption(f"active_session_id: {active_id[:8] if active_id else 'None'}")


def autoload_latest_session_if_exists(supabase: Any) -> None:
    if st.session_state.active_session_id:
        return
    sessions = fetch_sessions(supabase)
    if not sessions:
        return
    latest_id = str(sessions[0]["id"])
    st.session_state.active_session_id = latest_id
    st.session_state.last_loaded_session_id = latest_id
    st.session_state.chat_history = load_session_chat_history(supabase, latest_id)


def main() -> None:
    st.set_page_config(page_title=BOT_NAME, page_icon="📚", layout="wide")
    setup_logging()
    load_dotenv(PROJECT_ROOT / ".env", override=False)  # 로컬 .env 로드 (Streamlit Secrets는 override 안 함)
    ensure_session_state()
    apply_app_css()
    render_header()

    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not anon_key:
        st.error(
            "Supabase 설정이 필요합니다. Streamlit Cloud의 경우 "
            "Settings → Secrets에 `SUPABASE_URL`, `SUPABASE_ANON_KEY`를 등록해 주세요."
        )
        st.stop()

    # 클라이언트를 session_state에 캐시하여 로그인 세션 유지 (Streamlit rerun 시에도 유지)
    supabase = st.session_state.get("supabase")
    if not supabase:
        supabase = create_supabase_client()
        st.session_state.supabase = supabase

    render_api_keys_section()

    current_user = st.session_state.current_user
    if not current_user:
        try:
            res = supabase.auth.get_user()
            if res.user:
                current_user = res.user
                st.session_state.current_user = current_user
        except Exception:
            pass

    if not current_user:
        render_auth_section(supabase)
        st.info("챗봇을 사용하려면 로그인 또는 회원가입을 해 주세요.")
        st.stop()

    render_logged_in_sidebar(supabase, current_user)
    user_id = str(current_user.id)

    st.caption("Supabase에 세션/벡터를 저장하고, 스트리밍으로 답변을 생성합니다.")

    autoload_latest_session_if_exists(supabase)
    render_chat_history()
    render_sidebar(supabase, user_id)

    question = st.chat_input("질문을 입력해 주세요.")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        handle_user_question(supabase, user_id, question)
        st.rerun()


if __name__ == "__main__":
    main()
