/retrieve = query text → query embedding → vector search in pgvector → formatted retrieval results`

第一段：/retrieve API 入口
@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    q_vec = embed_queries([req.query_text])[0]
    rows = retrieve_top_k(q_vec, req.top_k, document_id=req.document_id)
    retrieved = citations_from_rows(rows)
    return {"ok": True, "query_text": req.query_text, "top_k": req.top_k, "results": retrieved}