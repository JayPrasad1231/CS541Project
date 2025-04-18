import os
from dotenv import load_dotenv, find_dotenv
from google import genai

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

genAIclient = genai.Client(api_key=GOOGLE_API_KEY)

def chat(message, type, LLM):

    if type == "Matching":
        pass
    elif type == "Comparing":
        pass
    else:
        pass

    if LLM == "Gemini":
        response = genAIclient.models.generate_content(
            model="gemini-2.0-flash", contents=message
        )
        return response
    
    elif LLM == "Mistral":
        pass
