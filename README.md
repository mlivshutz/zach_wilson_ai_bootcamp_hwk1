# Zach Wilson AI Bootcamp - Homework 1

## FastAPI Hello World Application

This is a simple FastAPI application that demonstrates basic API endpoints.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python main.py
   ```

3. **Or run with uvicorn directly:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

- **GET /** - Returns "Hello World" message
- **GET /hello/{name}** - Returns personalized greeting
- **GET /health** - Health check endpoint

## Testing

Visit these URLs in your browser:
- http://localhost:8000/ 
- http://localhost:8000/hello/YourName
- http://localhost:8000/health

## API Documentation

FastAPI automatically generates interactive API documentation:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Project Structure

```
zach_wilson_ai_bootcamp_hwk1/
├── main.py           # FastAPI application
├── requirements.txt  # Python dependencies
└── README.md        # This file
``` 