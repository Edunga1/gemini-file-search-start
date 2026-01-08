import os

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types


def require_api_key() -> str:
  api_key = os.environ.get("GENAI_API_KEY")
  if not api_key:
    st.error("환경 변수 GENAI_API_KEY를 설정해 주세요.")
    st.stop()
  return api_key


@st.cache_data(show_spinner=False, ttl=60)
def list_file_search_stores(api_key: str):
  client = genai.Client(api_key=api_key)
  stores = list(client.file_search_stores.list())
  return [
    {
      "선택": False,
      "표시 이름": getattr(store, "display_name", ""),
      "리소스 이름": store.name,
      "문서 수": getattr(store, "active_documents_count", ""),
      "생성 시각": str(getattr(store, "create_time", "")),
      "수정 시각": str(getattr(store, "update_time", "")),
    }
    for store in stores
  ]


def render_store_selector(stores: list[dict]) -> tuple[str | None, pd.DataFrame]:
  df = pd.DataFrame(stores)
  edited = st.data_editor(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
      "선택": st.column_config.CheckboxColumn("선택", default=False, width="small"),
    },
    disabled=["표시 이름", "리소스 이름", "문서 수", "생성 시각", "수정 시각"],
    key="stores_editor",
  )

  selected = edited[edited["선택"]]
  if len(selected) > 1:
    keep = selected.iloc[-1]["리소스 이름"]
    st.warning("스토어는 하나만 선택할 수 있습니다. 마지막으로 체크한 스토어만 적용됩니다.")
    edited.loc[:, "선택"] = edited["리소스 이름"] == keep
    return keep, edited
  if len(selected) == 1:
    return selected.iloc[0]["리소스 이름"], edited
  return None, edited


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


def main():
  st.set_page_config(page_title="Gemini FileSearch Stores", page_icon="📂")
  st.title("Gemini FileSearch Stores")
  st.caption("Gemini File Search 스토어를 선택해 쿼리합니다.")

  api_key = require_api_key()

  if st.button("새로 고침", type="secondary"):
    list_file_search_stores.clear()

  try:
    stores = list_file_search_stores(api_key)
  except Exception as exc:  # noqa: BLE001
    st.error(f"스토어 목록을 불러오는 중 오류가 발생했습니다: {exc}")
    st.stop()

  if not stores:
    st.info("등록된 FileSearch 스토어가 없습니다.")
    return

  st.metric("스토어 수", len(stores))
  selected_store, _ = render_store_selector(stores)

  st.subheader("프롬프트")
  prompt = st.text_area("질의 내용", height=160, placeholder="프롬프트를 입력하세요.")
  query_button = st.button(
    "질의 실행",
    type="primary",
    disabled=(selected_store is None) or (not prompt.strip()),
  )

  if query_button:
    with st.spinner("응답 생성 중..."):
      try:
        result = run_query(api_key, selected_store, prompt)
        st.session_state["last_result"] = result
      except Exception as exc:  # noqa: BLE001
        st.error(f"쿼리 수행 중 오류가 발생했습니다: {exc}")

  if "last_result" in st.session_state:
    st.subheader("결과")
    st.write(st.session_state["last_result"])


if __name__ == "__main__":
  main()
