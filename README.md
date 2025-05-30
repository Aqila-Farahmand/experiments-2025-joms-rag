# RAG System designed to assist patients with hypertension

This repository provides the implementation of a **Retrieval-Augmented Generation (RAG)** system designed to assist **hypertensive patients** by delivering context-aware, trustworthy, and privacy-preserving information. 
The system follows a modular pipeline, enabling focused evaluation and improvement at each stage.

---

## Project Overview

The goal of this project is to design and evaluate a RAG-based system that offers reliable support to individuals managing hypertension. 
The system integrates information retrieval with generative language models to improve accuracy and relevance, especially in a health-sensitive context.

Our methodology is rooted in two core principles:

- **Privacy Preservation**: Ensuring that no sensitive patient data is exposed.
- **Reliable Communication**: Generating medically appropriate, clear, and trustworthy responses in Italian Language.

---

## Methodology

The RAG pipeline is structured into five main phases:

1. **Data Preparation**  
   Medical documents, patient FAQs, and other hypertension-relevant texts are collected and preprocessed.  
   → [`documents/`](./documents)

2. **Embedding Generation & Chunk Evaluation**  
   Documents are segmented and converted into vector embeddings. Various chunking methods and embedding models are evaluated for performance.  
   → [`embedding/`](./embedding)

3. **Retrieval & Augmentation**  
   Given a user query, the system retrieves the most relevant text chunks and augments the prompt sent to the generator.  
   → [`rag/`](./rag)

4. **Response Generation**  
   Using a SLM, the system generates responses based on the retrieved context.  
   → [`generations/`](./generations)

5. **Evaluation**  
   We evaluate the RAG system using both automatic metrics and human assessments, and compare it to single-LM (SLM) baselines under different prompting strategies.  
   → [`evaluations/`](./evaluations)

---

## 📁 Repository Structure

├── documents/ # Data augmentation and preprocessing scripts
├── embedding/ # Embedding generation
├── analysis/ # Chunk size and overlap ratio analysis
├── rag/ # Retrieval logic and prompt refinements pipeline
├── generation/ # Interfaces for SLM-based response generation
├── evaluation/ # Evaluation scripts, performance metrics and plots 
├── pyproject.toml # project dependencies
└── README.md # Project documentation


---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Aqila-Farahmand/experiments-2025-joms-rag.git
cd experiments-2025-joms-rag
```
### 2. Install dependencies
```bash
 poetry install --no --root
```

### 3. Run the pipeline
Each module is designed to be run independently. 
Start with the documents module and progress through embedding, rag, generations and evaluations.
#### Example: generate embeddings
```bash
 python embedding/__main__.py
```

📊 Evaluation
Our evaluation includes:

+ Quantitative Metrics: G-eval metric

+ Qualitative Analysis: Human feedback on response helpfulness and safety

+ Comparative Baselines: Comparison against SLM-only responses using role-playing (zero shot) and full-context (few shots) prompts engineering techniques

🤝 Contributing
We welcome community contributions!If you’d like to help improve the system:

1. Fork the repo

2. Create a new branch 

3. Submit a pull request

4. Feel free to open issues for bug reports or feature suggestions.

📄 License
This project is licensed under the  Apache License. See the LICENSE file for details.

📬 Contact
For questions or collaborations, please contact [aqila.farahmand@uniurb.it].
