from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import tool
from openai import OpenAI
import json
import os

#Model
from langchain_openai import ChatOpenAI

response_llm = ChatOpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY"),
    model="gpt-oss-120b",
)

#Prompt template
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant answering questions about SWISS International Airlines, "
     "especially regarding sustainability and environmental initiatives.\n\n"
     "Use the following context from retrieved documents to answer the question. "
     "If you are unsure or the answer isn't in the context, say that you don't know.\n\n"
     "At the end of your answer, list the sources you used as bullet points (based on the provided context).\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

#RAG
from langchain_community.document_loaders import PyPDFLoader

# List of PDF file paths
pdf_files = [
    "LH-Factsheet-Sustainability-2024.pdf",
    "SWISS_Environmental_Report_2024_EN.pdf"
]

# Load all documents
all_pages_pdf = []
for pdf in pdf_files:
    loader = PyPDFLoader(pdf)
    pages = loader.load()
    all_pages_pdf.extend(pages)

print(f"Loaded {len(all_pages_pdf)} pages from {len(pdf_files)} pdf documents.")

#
from langchain_community.document_loaders import BSHTMLLoader
all_paths =     [
        "EU Regulation 261:04: your rights as a passenger | SWISS.html",
        "FAQs: bookings and fares | SWISS.html",
        "Refunds_SWISS.html",
        "Terms and conditions of use and legal information | SWISS.html"
    ]

all_pages_html = []

for path in all_paths:
    loader = BSHTMLLoader(path)
    html_doc = loader.load()
    all_pages_html.extend(html_doc)

print(f"Loaded {len(all_pages_html)} pages from {len(all_paths)} html documents.")

os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyLangChainBot/1.0; +https://example.com/bot)"
from langchain_community.document_loaders import WebBaseLoader


loader_multiple_pages = WebBaseLoader(
    ["https://www.bazl.admin.ch/bazl/en/home/passagiere/fluggastrechte.html",
     "https://www.bazl.admin.ch/bazl/en/home/passagiere/fluggastrechte/nichtbefoerderung--annullierung-und-grosse-verspaetungen/nichtbefoerderung--ueberbuchung-.html",
     "https://www.bazl.admin.ch/bazl/en/home/passagiere/fluggastrechte/nichtbefoerderung--annullierung-und-grosse-verspaetungen/annullierung-des-fluges.html",
     "https://www.bazl.admin.ch/bazl/en/home/passagiere/fluggastrechte/nichtbefoerderung--annullierung-und-grosse-verspaetungen/grosse-verspaetungen.html"
    ]
)


# Beispiel Swiss:

# loader_multiple_pages = WebBaseLoader(
#     ["https://www.swiss.com/in/en/terms-conditions/terms-conditions.html",
#      "https://www.swiss.com/in/en/terms-conditions/passenger-rights.html",
#      "https://www.swiss.com/in/en/customer-support/faq/booking-tariffs.html",
#      "https://www.swiss.com/in/en/customer-support/refunds"]
# )

websites = loader_multiple_pages.load()
print(f"Loaded {len(websites)} websites from {len(pdf_files)} documents.")

for i, doc in enumerate(websites):
    print(f"--- Document {i+1} ---")
    print("Source:", doc.metadata.get("source", "N/A"))
    print("Length (chars):", len(doc.page_content))
    print("Preview:", doc.page_content[:300].replace("\n", " "), "\n")

#Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# get both websites and pdfs together
all_docs = all_pages_pdf + all_pages_html + websites

# define the splitter and strategy
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = splitter.split_documents(all_docs)

import numpy as np

lengths = [len(s.page_content) for s in splits]
print(f"Initial documents: {len(all_docs)}")
print(f"Total chunks: {len(splits)}")
print(f"Avg length: {np.mean(lengths):.1f}")
print(f"Min: {np.min(lengths)}, Max: {np.max(lengths)}")

