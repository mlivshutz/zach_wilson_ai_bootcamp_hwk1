# Zach Wilson AI Bootcamp - Homework 1

## ChatGPT-Powered FastAPI Application

This is a FastAPI application with a beautiful chat interface that connects to ChatGPT for AI-powered conversations.

## Features

- 🤖 **ChatGPT Integration** - Real-time conversations with OpenAI's GPT-3.5-turbo
- 💬 **Beautiful Chat UI** - Modern, responsive chat interface
- ⚡ **FastAPI Backend** - High-performance async API
- 📱 **Mobile Responsive** - Works great on desktop and mobile
- 🎨 **Modern Design** - Gradient backgrounds and smooth animations

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env
```

Add your OpenAI API key to the `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

**To get an OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key and add it to your `.env` file

### 3. Run the Application
```bash
python main.py
```

Or run with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Chat Interface
Visit http://localhost:8000 in your browser to start chatting!

## API Endpoints

### Chat Endpoints
- **GET /** - Chat interface (HTML page)
- **POST /ask** - Send message to ChatGPT
  ```json
  {
    "message": "Hello, how are you?"
  }
  ```

### Other Endpoints
- **GET /api/hello** - Returns "Hello World" message
- **GET /hello/{name}** - Returns personalized greeting
- **GET /health** - Health check endpoint

## Testing the API

### Using the Web Interface
1. Open http://localhost:8000
2. Type a message and press Enter or click Send
3. Watch ChatGPT respond in real-time!

### Using curl
```bash
# Test ChatGPT endpoint
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is the meaning of life?"}'

# Test other endpoints
curl http://localhost:8000/api/hello
curl http://localhost:8000/hello/John
curl http://localhost:8000/health
```

## API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Project Structure

```
zach_wilson_ai_bootcamp_hwk1/
├── main.py              # FastAPI application with ChatGPT integration
├── static/
│   └── index.html       # Beautiful chat interface
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
└── README.md           # This file
```

## Chat Interface Features

- **Real-time messaging** with typing indicators
- **Error handling** with user-friendly messages
- **Responsive design** that works on all devices
- **Smooth animations** and modern styling
- **Auto-scroll** to latest messages
- **Enter key support** for quick messaging

## Environment Variables

Create a `.env` file with:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Security Notes

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure and private
- Monitor your OpenAI usage to avoid unexpected charges

## Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Make sure you created a `.env` file
   - Check that your API key is correctly formatted
   - Ensure the file is in the same directory as `main.py`

2. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Make sure you're in the correct directory

3. **Chat interface not loading**
   - Check that the `static/` directory exists
   - Verify `index.html` is in the `static/` folder
   - Try refreshing the browser

### API Rate Limits
OpenAI has rate limits on API usage. If you hit limits:
- Wait a few minutes before trying again
- Consider upgrading your OpenAI plan for higher limits
- Implement request throttling if needed

## Next Steps

- Add conversation history storage
- Implement user authentication
- Add support for different AI models
- Create conversation export functionality
- Add file upload capabilities 