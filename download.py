import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ['GENAI_API_KEY'])

page = client.file_search_stores.list()
store = page[0]

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="""vimscript에 대해 알려줘""",
    config=types.GenerateContentConfig(
        tools=[
            types.Tool(
                file_search=types.FileSearch(
                    file_search_store_names=[store.name]
                )
            )
        ]
    )
)

print(response.text)
