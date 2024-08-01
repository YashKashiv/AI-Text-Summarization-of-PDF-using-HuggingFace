# First we install required libraries
#pip install transformers PyPDF2 torch

import PyPDF2
from transformers import pipeline

def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page in range(num_pages):
            pdf_text += reader.pages[page].extract_text()
    return pdf_text

def summarize_text(text, max_length=150, min_length=30):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def save_summary_to_file(summary, output_path):
    with open(output_path, "w") as file:
        file.write(summary)

def summarize_pdf(pdf_path, output_path):
    text = extract_text_from_pdf(pdf_path)
    summary = summarize_text(text)
    save_summary_to_file(summary, output_path)

pdf_path = 'DEEP LEARNING.pdf'
output_path = 'summary.txt'
summarize_pdf(pdf_path, output_path)
print(f"Summary saved to {output_path}")