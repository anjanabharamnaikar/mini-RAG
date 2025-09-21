# Mini RAG + Reranker Sprint

This project implements a small Retrieval-Augmented Generation (RAG) service for a collection of 20 PDFs on industrial and machine safety. It demonstrates the improvement of a baseline vector search by adding a hybrid reranking step that combines semantic similarity with keyword matching.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/anjanabharamnaikar/mini-RAG
    cd mini-rag-project
    ```

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare Data:**
    -   Download and unzip the `industrial-safety-pdfs.zip` file.
    -   Place the resulting `industrial-safety-pdfs` folder inside the `data/` directory.
    -   Ensure `sources.json` is also inside the `data/` directory.

## How to Run

Follow these steps in order.

### Step 1: Ingest Data

Run the ingestion script. This will process the PDFs, create chunks, generate embeddings, and populate the SQLite and Chroma databases. This only needs to be done once.

```bash
python ingest.py
```
This process may take a few minutes as it downloads the embedding model and processes all 20 documents.

### Step 2: Start the API and frontend in 2 seperate terminals

Launch the FastAPI service using Uvicorn.

```bash
uvicorn api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

```bash
streamlit run app.py
```
the frontend will be available at `http://192.168.1.3:8501`

### Step 3: Query the API (Example `curl` Requests)

You can now send questions to the `/ask` endpoint.

**Easy Example (Baseline Mode):**
```bash
curl -X POST "[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)" \
-H "Content-Type: application/json" \
-d '{
  "q": "What is machine guarding?",
  "k": 3,
  "mode": "baseline"
}'
```

**Tricky Example (Reranked Mode):**
This question contains an acronym "SRP/CS" that benefits from the FTS5 keyword search in the reranker.
```bash
curl -X POST "[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)" \
-H "Content-Type: application/json" \
-d '{
  "q": "What does SRP/CS stand for?",
  "k": 3,
  "mode": "reranked"
}'
```

### Step 4: Run Evaluation

To see the before/after comparison, run the evaluation script while the API is running in another terminal.

```bash
python evaluate.py
```

## Results Table

The `evaluate.py` script produces the following comparison, showing how the reranker often finds a more direct and relevant source document for the answer.

| Question                                                              | Baseline Answer (Top Chunk)                                                                                  | Baseline Source                                                 | Reranked Answer (Top Chunk)                                                                                  | Reranked Source                                                   |
|:----------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|
| What is machine guarding?                                             | Safeguarding Equipment and Protecting Employees from Amputations. This informational booklet provides a g... | 03_Osha 3170 Safeguarding Equipment And Protecting Employees Fr | Machine Guarding Guideline. August 2023. Revision 2.0. SLAC-I-730-0A13J-001-R2.0. Verified and Releas... | 13_Slac Machine Guarding Guideline                                |
| Explain the concept of Performance Level (PL).                        | Safety-related characteristics of control systems. In Europe, the safety of machinery is regulated by th... | 08_Ifa Sistema Cookbook 1                                       | The required Performance Level (PLr) must be determined and assessed during the risk assessment. The PL... | 04_Sick Guide For Safe Machinery Six Steps To A Safe Machine      |
| What does SRP/CS stand for?                                           | 5.2.2 Subsystems. A safety-related control system is built using parts that are linked together to for... | 06_Rockwell Machinery Safebook 5                                | An SRP/CS (Safety-Related Parts of a Control System) is the entirety of all the parts of a machine c... | 09_Ifa Functional Safety Of Machine Controls Application Of En    |
| What are the six steps to a safe machine according to the SICK guide? | Guide for Safe Machinery. Six steps to a safe machine | SICK | USA. SIX STEPS TO A SAFE MACHINE. For new, existing or modified machinery in accordance with global safe... | 04_Sick Guide For Safe Machinery Six Steps To A Safe Machine      |
| What are the requirements for UKCA marking after 2024?                | **ABSTAINED**: Top result score (0.43) is below the threshold of 0.4.                                           | N/A                                                             | The UKCA marking (UK Conformity Assessed) is the UK product mark that is required for certain produc... | 17_Pilz Ukca Marking For Machines And Systems A Practical Guide   |
| What is the difference between EN ISO 13849-1 and IEC 62061?          | EN ISO 13849-1 and EN 62061 – A Comparison. Introduction. From a historical point of view, the two m... | 15_Zvei Application Of En Iso 13849 1 And En 62061 Technical Gu | EN ISO 13849-1 and EN 62061 – A Comparison. Introduction. From a historical point of view, the two m... | 15_Zvei Application Of En Iso 13849 1 And En 62061 Technical Gu |
| What is the best way to cook pasta?                                   | **ABSTAINED**: Top result score (0.05) is below the threshold of 0.4.                                           | N/A                                                             | **ABSTAINED**: Top result score (0.24) is below the threshold of 0.4.                                           | N/A                                                               |
| What is SISTEMA?                                                      | Introduction. SISTEMA (Safety Integrity Software Tool for the Evaluation of Machine Applications) is a... | 08_Ifa Sistema Cookbook 1                                       | Introduction. SISTEMA (Safety Integrity Software Tool for the Evaluation of Machine Applications) is a... | 08_Ifa Sistema Cookbook 1                                       |

