import os
import tempfile
import time

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types


def require_api_key() -> str:
  api_key = os.environ.get("GENAI_API_KEY")
  if not api_key:
    st.error("환경 변수 GENAI_API_KEY를 설정해 주세요.")
    st.stop()
  assert api_key
  return api_key


@st.cache_data(show_spinner=False, ttl=60)
def list_file_search_stores(api_key: str):
  client = genai.Client(api_key=api_key)
  stores = list(client.file_search_stores.list())
  return [
    {
      "표시 이름": getattr(store, "display_name", ""),
      "리소스 이름": store.name,
      "문서 수": getattr(store, "active_documents_count", getattr(store, "activeDocumentsCount", "")),
      "생성 시각": str(getattr(store, "create_time", "")),
      "수정 시각": str(getattr(store, "update_time", "")),
    }
    for store in stores
  ]


PAGE_SIZE = 20


@st.cache_data(show_spinner=False, ttl=60)
def list_documents(api_key: str, store_name: str, page_token: str | None = None):
  client = genai.Client(api_key=api_key)
  config: types.ListDocumentsConfigOrDict = {"page_size": PAGE_SIZE}
  if page_token:
    config["page_token"] = page_token
  pager = client.file_search_stores.documents.list(
    parent=store_name,
    config=config,
  )
  docs = []
  for doc in pager:
    docs.append({
      "문서 이름": doc.name,
      "표시 이름": getattr(doc, "display_name", ""),
      "크기": getattr(doc, "size_bytes", ""),
      "생성 시각": str(getattr(doc, "create_time", "")),
      "수정 시각": str(getattr(doc, "update_time", "")),
    })
    if len(docs) >= PAGE_SIZE:
      break
  next_page_token = pager._config.get("page_token") if hasattr(pager, "_config") else None
  # 입력한 토큰과 같으면 다음 페이지 없음
  if next_page_token == page_token:
    next_page_token = None
  return docs, next_page_token


def upload_document_to_store(api_key: str, store_name: str, uploaded_file) -> None:
  client = genai.Client(api_key=api_key)
  suffix = os.path.splitext(uploaded_file.name)[1]
  with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded_file.getbuffer())
    tmp_path = tmp.name

  try:
    operation = client.file_search_stores.upload_to_file_search_store(
      file=tmp_path,
      file_search_store_name=store_name,
      config={"display_name": uploaded_file.name},
    )
    while not operation.done:
      time.sleep(2)
      operation = client.operations.get(operation)
  finally:
    os.remove(tmp_path)


def render_store_selector(stores: list[dict]) -> str | None:
  df = pd.DataFrame(stores)
  selector = st.dataframe(
    df,
    hide_index=True,
    width="stretch",
    selection_mode="single-row",
    on_select="rerun",
    key="stores_selector",
  )

  selection = getattr(selector, "selection", {}) or {}
  rows = selection.get("rows", [])
  if rows:
    idx = rows[0]
    return df.iloc[idx]["리소스 이름"]
  return None


def run_query(api_key: str, store_name: str, history: list[dict]) -> str | None:
  client = genai.Client(api_key=api_key)
  contents: types.ContentListUnionDict = [
    types.Content(role=msg["role"], parts=[types.Part(text=msg["text"])])
    for msg in history
  ]
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config=types.GenerateContentConfig(
      tools=[
        types.Tool(
          file_search=types.FileSearch(
            file_search_store_names=[store_name],
          )
        )
      ]
    ),
  )
  return response.text


def render_page_header():
  st.set_page_config(page_title="Gemini FileSearch Stores", page_icon="📂")
  st.title("Gemini FileSearch Stores")
  st.caption("Gemini File Search 스토어를 선택해 쿼리합니다.")


def reset_docs_pagination():
  keys_to_remove = [k for k in st.session_state if k.startswith("docs_page_")]
  for k in keys_to_remove:
    del st.session_state[k]


def render_refresh_controls():
  if st.button("새로 고침", type="secondary"):
    list_file_search_stores.clear()
    list_documents.clear()
    reset_docs_pagination()

def load_stores(api_key: str) -> list[dict] | None:
  try:
    return list_file_search_stores(api_key)
  except Exception as exc:
    st.error(f"스토어 목록을 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()


def get_docs_page_state(store_name: str) -> dict:
  key = f"docs_page_{store_name}"
  if key not in st.session_state:
    st.session_state[key] = {"page_tokens": [None], "current_index": 0}
  return st.session_state[key]


def render_documents_section(api_key: str, selected_store: str):
  st.subheader("문서 목록")

  page_state = get_docs_page_state(selected_store)
  current_index = page_state["current_index"]
  page_tokens = page_state["page_tokens"]
  current_token = page_tokens[current_index]

  try:
    docs, next_page_token = list_documents(api_key, selected_store, current_token)
  except Exception as exc:
    st.error(f"문서 목록을 불러오는 중 오류가 발생했습니다: {exc}")
    docs, next_page_token = [], None

  if docs:
    docs_df = pd.DataFrame(docs)
    docs_selector = st.dataframe(
      docs_df,
      hide_index=True,
      width="stretch",
      selection_mode="multi-row",
      on_select="rerun",
      key=f"docs_selector_{selected_store}_{current_index}",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
      if st.button("◀ 이전", disabled=(current_index == 0), key=f"prev_{selected_store}"):
        page_state["current_index"] = current_index - 1
        st.rerun()
    with col2:
      st.write(f"페이지 {current_index + 1}")
    with col3:
      if st.button("다음 ▶", disabled=(next_page_token is None), key=f"next_{selected_store}"):
        if len(page_tokens) <= current_index + 1:
          page_tokens.append(next_page_token)
        page_state["current_index"] = current_index + 1
        st.rerun()

    selection = getattr(docs_selector, "selection", {}) or {}
    selected_rows = selection.get("rows", [])
    if selected_rows:
      st.warning("선택한 문서를 삭제하면 되돌릴 수 없습니다.", icon="⚠️")
      if st.button("선택 문서 삭제", type="primary", key=f"docs_delete_{selected_store}"):
        client = genai.Client(api_key=api_key)
        for idx in selected_rows:
          row = docs_df.iloc[idx]
          doc_name = row.get("문서 이름")
          if not doc_name:
            continue
          with st.spinner(f"삭제 중: {doc_name}"):
            try:
              client.file_search_stores.documents.delete(name=doc_name, config={"force": True})
              st.toast(f"삭제 완료: {doc_name}")
            except Exception as exc:
              st.error(f"삭제 실패 ({doc_name}): {exc}")
        list_documents.clear()
        page_state["page_tokens"] = [None]
        page_state["current_index"] = 0
        st.rerun()
  else:
    st.info("선택한 스토어에 문서가 없습니다.")


def render_upload_section(api_key: str, selected_store: str):
  st.subheader("파일 업로드")
  uploaded_files = st.file_uploader(
    "파일을 드래그 앤 드롭해서 업로드하세요.",
    accept_multiple_files=True,
    key="file_uploader",
  )
  if uploaded_files:
    processed = st.session_state.setdefault("uploaded_files", {})
    for uploaded_file in uploaded_files:
      file_key = f"{selected_store}:{uploaded_file.name}:{uploaded_file.size}"
      if processed.get(file_key):
        st.info(f"이미 업로드된 파일: {uploaded_file.name}")
        continue
      with st.spinner(f"{uploaded_file.name} 업로드 중..."):
        try:
          upload_document_to_store(api_key, selected_store, uploaded_file)
          processed[file_key] = True
          list_documents.clear()
          reset_docs_pagination()
          st.success(f"업로드 완료: {uploaded_file.name}")
        except Exception as exc:
          st.error(f"파일 업로드 중 오류가 발생했습니다: {exc}")


def get_chat_history(store_name: str) -> list[dict]:
  key = f"chat_history_{store_name}"
  if key not in st.session_state:
    st.session_state[key] = []
  return st.session_state[key]


def render_chat(api_key: str, selected_store: str | None):
  st.subheader("대화")

  if selected_store:
    history = get_chat_history(selected_store)

    # 대화 기록 표시
    for msg in history:
      # Gemini는 "model", Streamlit은 "assistant" 사용
      role = "assistant" if msg["role"] == "model" else msg["role"]
      with st.chat_message(role):
        st.write(msg["text"])

    # 새 대화 버튼
    if history and st.button("대화 초기화", type="secondary"):
      history.clear()
      st.rerun()

  prompt = st.chat_input("메시지를 입력하세요.", disabled=(selected_store is None))

  if prompt and selected_store is not None:
    history = get_chat_history(selected_store)
    history.append({"role": "user", "text": prompt})

    with st.chat_message("user"):
      st.write(prompt)

    with st.chat_message("assistant"):
      with st.spinner("응답 생성 중..."):
        try:
          result = run_query(api_key, selected_store, history)
          history.append({"role": "model", "text": result})
          st.write(result)
        except Exception as exc:
          st.error(f"쿼리 수행 중 오류가 발생했습니다: {exc}")
          history.pop()  # 실패한 user 메시지 제거


def main():
  render_page_header()

  api_key = require_api_key()
  render_refresh_controls()

  stores = load_stores(api_key)

  if not stores:
    st.info("등록된 FileSearch 스토어가 없습니다.")
    return

  st.metric("스토어 수", len(stores))
  selected_store = render_store_selector(stores)

  if selected_store:
    render_documents_section(api_key, selected_store)
    render_upload_section(api_key, selected_store)

  render_chat(api_key, selected_store)


if __name__ == "__main__":
  main()
