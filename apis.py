from flask import Flask, request, jsonify
from flask_cors import CORS
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from googletrans import Translator
import PyPDF2
import docx
import time
import re
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Abstractive Summarization Model (fine-tuned BART)
config = PeftConfig.from_pretrained("hammadali1805/legal_bart_large_cnn")
base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
abs_model = PeftModel.from_pretrained(base_model, "hammadali1805/legal_bart_large_cnn")
abs_tokenizer = AutoTokenizer.from_pretrained("hammadali1805/legal_bart_large_cnn")

# Lightweight LLM for framing extractive summary and chat (using distilgpt2)
llm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
llm_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

translator = Translator()

# Global variable to store the current document context for query operations
current_context = ""

# Load the Keras sentence scoring model (assumed to take raw text as input)
scoring_model = load_model('sentence_scoring_model.keras')

def score_sentence(sentence):
    """
    Uses the Keras model to score a sentence.
    The model is assumed to output a score (float) for each sentence.
    """
    # The model is assumed to accept a list of sentences.
    # The output is assumed to be a list/array with one score per sentence.
    score = scoring_model.predict([sentence])[0][0]
    return score

def summarize_abstractive(text, max_length=512, min_length=30, num_beams=2):
    """
    Generate an abstractive summary using a fine-tuned BART model.
    """
    input_ids = abs_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = abs_model.base_model.generate(
        input_ids,
        num_beams=num_beams,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        early_stopping=True
    )
    summary = abs_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def score_and_select_sentences(text, num_sentences=3):
    """
    Splits text into sentences, scores them using the imported Keras model,
    and returns the top-scoring sentences.
    """
    # A simple sentence splitter using regular expressions
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= num_sentences:
        return sentences
    # Score each sentence using the Keras model
    scored = [(sentence, score_sentence(sentence)) for sentence in sentences]
    # Sort sentences by score in descending order and pick the top ones
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored[:num_sentences]]
    return top_sentences

def frame_extractive_summary(sentences):
    """
    Uses a lightweight LLM (distilgpt2) to frame the selected sentences into a coherent extractive summary.
    """
    prompt = "Frame a coherent extractive summary from the following key points:\n" + "\n".join(sentences) + "\nSummary:"
    input_ids = llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=256)
    output_ids = llm_model.generate(input_ids, max_length=256, do_sample=True, temperature=0.7)
    summary = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    summary = summary.replace(prompt, "").strip()
    return summary

def summarize_extractive(text, num_sentences=3):
    """
    Implements extractive summarization by scoring sentences with a Keras model,
    selecting the highest scoring ones, and framing them using a lightweight LLM.
    """
    top_sentences = score_and_select_sentences(text, num_sentences=num_sentences)
    summary = frame_extractive_summary(top_sentences)
    return summary

def chat_with_llm(query):
    """
    A simple chat function that uses a lightweight LLM (distilgpt2) to answer queries based on the stored document context.
    """
    global current_context
    prompt = f"You are an assistant. Based on the following document context, answer the query concisely.\n\nDocument Context:\n{current_context}\n\nQuery: {query}\nAnswer:"
    input_ids = llm_tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = llm_model.generate(input_ids, max_length=150, do_sample=True, temperature=0.7)
    response = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

@app.route('/summarizetext', methods=['POST'])
def summarize_text():
    global current_context
    data = request.json
    if 'text' not in data or 'language' not in data or 'checkbox' not in data:
        return jsonify({'error': 'Missing text, language, or checkbox in the request'}), 400

    text = data['text']
    language = data['language']
    checkbox = data['checkbox']  # Checkbox indicates whether to use abstractive summarization

    # Save the current context for query operations
    current_context = text

    # Use abstractive summarization if checkbox is true; otherwise, use the extractive summarizer.
    if checkbox:
        summary = summarize_abstractive(text)
    else:
        summary = summarize_extractive(text)

    translated_summary = translator.translate(summary, dest=language).text
    return jsonify({'summary': translated_summary})

@app.route('/summarisedoc', methods=['POST'])
def summarize_doc():
    global current_context
    if 'file' not in request.files or 'language' not in request.form or 'checkbox' not in request.form:
        return jsonify({'error': 'No file, language, or checkbox provided'}), 400

    file = request.files['file']
    language = request.form['language']
    checkbox = request.form['checkbox']

    # Extract text from file based on its extension
    if file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        text = ''.join([page.extract_text() for page in reader.pages])
    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        text = '\n'.join([para.text for para in doc.paragraphs])
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    # Save the current context for query operations
    current_context = text

    if checkbox == 'true':
        summary = summarize_abstractive(text)
    else:
        summary = summarize_extractive(text)

    translated_summary = translator.translate(summary, dest=language).text
    return jsonify({'summary': translated_summary})

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if 'text' not in data or 'language' not in data:
        return jsonify({'error': 'Missing text or language in the request'}), 400

    query_text = data['text']
    language = data['language']

    response_text = chat_with_llm(query_text)
    translated_response = translator.translate(response_text, dest=language).text
    time.sleep(2)
    return jsonify({'response': translated_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
