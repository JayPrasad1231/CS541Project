import os
from dotenv import load_dotenv, find_dotenv
from google import genai

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

client = genai.Client(api_key=GOOGLE_API_KEY)

# retrieve type and chat message from API

type = "Matching"
prompt = "" # retrieve from API
if type == "Matching":

    message = f"""

    Say if these two entites match: 

    {prompt}

    """

elif type == "Comparing":

    message = f"""

    Determine if entity A or B matches with the anchor record:

    {prompt}

    """

else:

    message = f"""

    Determine if entity A or B matches with the anchor record:

    {prompt}

    """


response = client.models.generate_content(
    model="gemini-2.0-flash", contents=message
)

# for idx, candidate in enumerate(response.candidates):
#     print(f"Response {idx + 1}: {candidate.content.response}")

print(response.text)
