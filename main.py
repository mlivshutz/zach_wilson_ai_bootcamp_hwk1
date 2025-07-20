from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create FastAPI instance
app = FastAPI(title="ChatGPT Chat API", version="1.0.0")

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        with open("static/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html>
            <body>
                <h1>Chat interface not found</h1>
                <p>Please make sure index.html exists in the static directory.</p>
            </body>
        </html>
        """

# Simple Hello World endpoint
@app.get("/api/hello")
async def read_root():
    return {"message": "Hello World"}

# ChatGPT endpoint
@app.post("/ask", response_model=ChatResponse)
async def ask_chatgpt(chat_message: ChatMessage):
    try:
        # Make request to OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": chat_message.message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract the response
        gpt_response = response.choices[0].message.content.strip()
        return ChatResponse(response=gpt_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with ChatGPT: {str(e)}")

# Additional endpoint with path parameter
@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)