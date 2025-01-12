from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from groq import Groq
import json

# Load environment variables
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# FastAPI app
app = FastAPI()

# Define the input schema for requests
class AIModelInput(BaseModel):
    input: str
    model: str
    temperature: float
    top_p: float

@app.get("/")
async def health_check():
    return {
        "check": "Health check is successfull!",
        "message": "Welcome to the SURD AI Model"
    }

@app.post("/ai-model")
def ai_model(payload: AIModelInput):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are an Expert agent in identifying actions, and entities to perform the actions on, from the given natural language input 
                        for the game characters. You identify the appropriate action (a one word) from the sentence, and the entity. You also generate 
                        a response based on the character's personality in natural language. The input you will be given will consist of the available 
                        actions for the character, the entities/objects in the environment for the character, a natural language input, and a 
                        personality description or backstory of the character. Based on the inputs, you identify appropriate action, preferably from 
                        the action list, the entity, and a response. You output in JSON with an array of actions, entities, and a natural language 
                        response. The response output should be precise and a short sentence, and output a JSON only. This is an example response: "{
                        "AI Response": {
                            "actions": [
                            "Shoot"
                            ],
                            "entities": [
                            "Enemy 1"
                            ],
                            "Verbal Response": "Target acquired, firing!"
                          }
                        }"
                    """
                },
                {
                    "role": "user",
                    "content": payload.input
                }
            ],
            model=payload.model,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        response = chat_completion.choices[0].message.content

        # Remove backticks
        cleaned_json = response.replace("```", "").replace("json", "")

        # Parse the valid JSON
        output = json.loads(cleaned_json)
        return {"response": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
