flowchart LR
  A["User/Client"] --> B["Retriever"]
  B --> E["Pinecone (optional)"]
  B --> F["BM25 / OpenSearch"]
  E --> C["Top-K Chunks"]
  F --> C
  C --> D["LLM (Prompt + Context)"]
  D --> G["Answer"]