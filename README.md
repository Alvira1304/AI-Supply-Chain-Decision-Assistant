# AI Supply Chain Decision Assistant

## Project Overview
The AI Supply Chain Decision Assistant is a Generative AI-powered application that helps businesses analyze inventory, identify stock risks, and make better purchasing decisions.

It uses Retrieval-Augmented Generation (RAG) to answer questions based on purchasing policy documents and current inventory data.

## Features
- AI-powered chatbot for supply chain questions.
- Inventory analysis and stock risk detection.
- Identification of products requiring reordering.
- Suggested order quantity calculation.
- RAG-based purchasing policy question answering.
- Supplier lead-time analysis.
- Downloadable inventory analysis reports in CSV format.

## Technologies Used
- Python
- Generative AI (LLM)
- Retrieval-Augmented Generation (RAG)
- Groq API
- ChromaDB
- Sentence Transformers
- Streamlit
- Pandas

## Project Structure

```text
AI_Supply_Chain_Decision_Assistant/
│
├── app.py
├── app_backup.py
├── build_knowledge_base.py
├── test_groq.py
├── test_retrieval.py
├── inventory.csv
├── requirements.txt
├── .gitignore
├── .env
├── chroma_db/
│
└── documents/
    └── purchasing_policy.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Alvira1304/AI-Supply-Chain-Decision-Assistant.git
```

2. Navigate to the project folder:

```bash
cd AI-Supply-Chain-Decision-Assistant
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create a `.env` file and add your Groq API key:

```text
GROQ_API_KEY=your_groq_api_key
```

5. Build the knowledge base if it does not already exist:

```bash
python build_knowledge_base.py
```

## Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## How RAG Works

1. The user enters a question.
2. The embedding model converts the question into numerical vectors.
3. ChromaDB retrieves relevant information from purchasing policy documents.
4. The retrieved information is passed to the Groq LLM.
5. The AI generates an answer using the retrieved context and available inventory data.

## Project Repository

GitHub: https://github.com/Alvira1304/AI-Supply-Chain-Decision-Assistant

## Note
This project is a decision-support prototype. AI-generated purchasing recommendations should be reviewed and approved by an authorized employee before placing orders.