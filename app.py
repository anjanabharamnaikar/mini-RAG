import streamlit as st
import requests

st.title("Mini RAG Q&A Service")

query = st.text_input("Enter your question:")
mode = st.radio("Search mode:", ["baseline", "reranked"])
k = st.slider("Number of results", 1, 10, 5)

if st.button("Ask"):
    response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={"q": query, "k": k, "mode": mode}
    )
    if response.status_code == 200:
        data = response.json()

        if data["answer"]:
            st.subheader("Answer")
            st.write(data["answer"])
        else:
            st.warning(f"Abstained: {data['abstain_reason']}")

        st.subheader("Retrieved Contexts")
        for ctx in data["contexts"]:
            st.markdown(f"**{ctx['title']}** (score: {ctx['score']:.2f})")
            st.write(ctx["content"])
            st.write("---")
    else:
        st.error(f"API Error: {response.text}")
