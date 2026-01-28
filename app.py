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


@st.cache_data(show_spinner=False, ttl=60)
def list_documents(api_key: str, store_name: str):
  client = genai.Client(api_key=api_key)
  docs = list(
    client.file_search_stores.documents.list(
      parent=store_name,
    )
  )
  return [
    {
      "문서 이름": doc.name,
      "표시 이름": getattr(doc, "display_name", ""),
      "크기": getattr(doc, "size_bytes", ""),
      "생성 시각": str(getattr(doc, "create_time", "")),
      "수정 시각": str(getattr(doc, "update_time", "")),
    }
    for doc in docs
  ]


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


def run_query(api_key: str, store_name: str, prompt: str) -> str:
  client = genai.Client(api_key=api_key)
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
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


def render_refresh_controls():
  if st.button("새로 고침", type="secondary"):
    list_file_search_stores.clear()
    list_documents.clear()

def load_stores(api_key: str) -> list[dict] | None:
  try:
    return list_file_search_stores(api_key)
  except Exception as exc:  # noqa: BLE001
    st.error(f"스토어 목록을 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()


def render_documents_section(api_key: str, selected_store: str):
  st.subheader("문서 목록")
  try:
    docs = list_documents(api_key, selected_store)
  except Exception as exc:
    st.error(f"문서 목록을 불러오는 중 오류가 발생했습니다: {exc}")
    docs = []
  if docs:
    docs_df = pd.DataFrame(docs)
    docs_selector = st.dataframe(
      docs_df,
      hide_index=True,
      width="stretch",
      selection_mode="multi-row",
      on_select="rerun",
      key=f"docs_selector_{selected_store}",
    )
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
            except Exception as exc:  # noqa: BLE001
              st.error(f"삭제 실패 ({doc_name}): {exc}")
        list_documents.clear()
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
          st.success(f"업로드 완료: {uploaded_file.name}")
        except Exception as exc:  # noqa: BLE001
          st.error(f"파일 업로드 중 오류가 발생했습니다: {exc}")


def render_prompt_and_results(api_key: str, selected_store: str | None):
  st.subheader("프롬프트")
  prompt = st.text_area("질의 내용", height=160, placeholder="프롬프트를 입력하세요.")
  query_button = st.button(
    "질의 실행",
    type="primary",
    disabled=(selected_store is None) or (not prompt.strip()),
  )

  if query_button and selected_store is not None:
    with st.spinner("응답 생성 중..."):
      try:
        result = run_query(api_key, selected_store, prompt)
        st.session_state["last_result"] = result
      except Exception as exc:
        st.error(f"쿼리 수행 중 오류가 발생했습니다: {exc}")

  if "last_result" in st.session_state:
    st.subheader("결과")
    st.write(st.session_state["last_result"])


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

  render_prompt_and_results(api_key, selected_store)


if __name__ == "__main__":
  main()
