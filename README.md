# 📜 LUFY - Law Understandable For You

LUFY is a platform that simplifies the process of summarizing legal documents, making legal content accessible to a broader audience. It allows users to upload legal documents in various formats and generates summaries in local languages like Hindi, Gujarati, Marathi, and more. Additionally, LUFY enables users to query specific sections of documents using a lightweight LLM chatbot, promoting better understanding and accessibility for non-English speakers.

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Tech Stack](#tech-stack)  
4. [Setup and Installation](#setup-and-installation)  
5. [Usage Instructions](#usage-instructions)  
6. [API Endpoints](#api-endpoints)  
7. [Frontend Overview](#frontend-overview)  
8. [Backend Overview](#backend-overview)  
9. [Future Improvements](#future-improvements)  
10. [License](#license)  

---

## 🌟 Project Overview

**LUFY** provides an intuitive platform for simplifying complex legal documents into concise summaries and supporting multiple languages. The platform serves legal professionals, clients, and anyone seeking clarity on legal jargon by providing **abstractive and extractive summaries**, improving accessibility to legal content.

Users can:
- Upload documents in PDF, TXT, DOCX formats.  
- Receive summarized content in multiple local languages.  
- Use a chatbot to query documents in natural language.  
- Get sentence-level importance scores using a regression model for extractive summarization.

---

## 💡 Key Features

- **Abstractive & Extractive Summaries**: Uses advanced models for high-quality summaries.
- **Sentence Scoring via ML**: A Keras regression model trained on legal data scores each sentence from 0 to 1 based on importance.
- **Multi-Language Support**: Summaries can be translated to Hindi, Gujarati, Marathi, and more.
- **Query Understanding via LLM**: Lightweight LLM-based chatbot enables smart document Q&A.
- **File Format Support**: Accepts PDF, TXT, DOCX files.
- **Lightweight Architecture**: Designed to run on low-resource environments.

---

## 🛠 Tech Stack

- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Python, Flask, Hugging Face Transformers, PEFT, TensorFlow  
- **Models**:
  - Facebook BART (abstractive summarization)
  - Custom Keras regression model (sentence importance scoring)
  - Lightweight open-source LLM (for chatbot query responses)
- **Libraries**:
  - `Flask`, `PyPDF2`, `python-docx`, `googletrans`
  - `transformers`, `peft`, `sentence-transformers`, `TensorFlow/Keras`

---

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.9.11 
- Flask  
- TensorFlow  
- Hugging Face Transformers + PEFT  
- Google Translate API Access  

### Installation Steps

```bash
git clone https://github.com/hammadali1805/CTRL-ALT-DEFEAT_LUFY_HACKNOVATE_6.git
cd CTRL-ALT-DEFEAT_LUFY_HACKNOVATE_6
pip install -r requirements.txt
```

### Run the Application

```bash
python apis.py
```

Access frontend by opening `index.html` in your browser.

---

## 💻 Usage Instructions

### Frontend

- **Upload Legal Files**: Upload files in PDF, DOCX, or TXT format.  
- **Choose Language**: Select output language for summary.  
- **Summarize**: Click Summarize to generate content.
- **Ask Questions**: Use the chatbot to query document contents in simple language.

### Backend

- Handles uploads, summarization, sentence scoring, translation, and chatbot Q&A.

---

## 🔌 API Endpoints

### 1. **Summarize Text**
- `POST /summarizetext`  
- Params: `text`, `language`  
- Returns: Translated summary

### 2. **Summarize Document**
- `POST /summarisedoc`  
- Params: `file`, `language`, `checkbox` (extractive/abstractive)  
- Returns: Summarized content

### 3. **Query Document**
- `POST /query`  
- Params: `language`, `question`  
- Returns: Chatbot-generated answer using lightweight LLM

### 4. **Score Sentences (Internal Use)**
- Automatically scores each sentence using a trained Keras regression model (`sentence_scoring_model.keras`) during extractive summary generation.

---

## 🎨 Frontend Overview

Built with basic HTML/CSS for ease of access.  
- **Upload Area**
- **Language Selector**
- **Chatbot Interface**

---

## 🛠 Backend Overview

- **Summarizer**: Fine-tuned BART model with legal domain adaptation.
- **Scoring Engine**: Custom-trained regression model that predicts importance score (0-1) for each sentence.
- **Translator**: Google Translate API.
- **Chatbot**: Fast, lightweight open-source LLM integrated with document context.

---

## 🚀 Future Improvements

- Improve scoring model using ROUGE/BERTScore for finer ground truths.  
- Replace chatbot with a retrieval-augmented system.  
- Add OCR support for scanned legal documents.  
- Deploy models with ONNX or TensorFlow Lite for mobile support.

---

## 📜 License

Licensed under the MIT License.
