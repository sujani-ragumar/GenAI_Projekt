from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import tool
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CEREBRAS_API_KEY")

if api_key:
    print("✅ API-Key wurde gefunden!")
    print("Erste 5 Zeichen:", api_key[:5], "...")
else:
    print("❌ Kein API-Key gefunden.")