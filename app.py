from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification
import re
from datetime import datetime

app = FastAPI()

# Mount the static directory to serve CSS and JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up the templates directory for HTML files
templates = Jinja2Templates(directory="templates")

# Track initialization status
initialization_status = {"initialized": False}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder for models and tokenizers
summarization_model = None
summarization_tokenizer = None
qa_model = None
qa_tokenizer = None
el_model = None
el_tokenizer = None
ner_model = None
ner_tokenizer = None

@app.on_event("startup")
async def load_models():
    global summarization_model, summarization_tokenizer, qa_model, qa_tokenizer, initialization_status
    try:
        # Load models (replace with your actual paths)
        summarization_model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/summarization_model").to(device)
        summarization_tokenizer = AutoTokenizer.from_pretrained("saved_models/summarization_model")

        qa_model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/qa_model").to(device)
        qa_tokenizer = AutoTokenizer.from_pretrained("saved_models/qa_model")

        el_model = AutoModelForSeq2SeqLM.from_pretrained("saved_models/el_model").to(device)
        el_tokenizer = AutoTokenizer.from_pretrained("saved_models/el_model")

        ner_model = AutoModelForTokenClassification.from_pretrained("saved_models/ner_model").to(device)
        ner_tokenizer = AutoTokenizer.from_pretrained("saved_models/ner_model")

        initialization_status["initialized"] = True
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")

class InputData(BaseModel):
    input: str

def truncate_to_complete_sentence(text):
    """Truncate the text to the last complete sentence."""
    match = re.search(r'^(.*?)([.!?])\s+[^.!?]*$', text)
    if match:
        return match.group(1) + match.group(2)
    return text

def capitalize_after_eos(text):
    """Capitalize the first letter of each sentence."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [sentence.capitalize() for sentence in sentences]
    return ' '.join(sentences)

def format_response(response):
    """Format the response by replacing certain phrases or capitalizing sentences."""
    # Replace "Chat Doctor" with "Wise Well"
    response = re.sub(r'\bChat Doctor\b', 'Wise Well', response, flags=re.IGNORECASE)
    
    # Apply truncation and capitalization
    response = truncate_to_complete_sentence(response)
    response = capitalize_after_eos(response)
    
    return response

def get_current_time():
    return datetime.now().strftime("%H:%M")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serve the main chatbot interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(data: InputData):
    if not initialization_status["initialized"]:
        return {"response": "The chatbot is still initializing. Please try again in a moment."}

    try:
        # Determine whether to summarize or answer a question based on input length
        input_tokens = qa_tokenizer.tokenize(data.input)

        if len(input_tokens) > 50:  # Threshold for deciding summarization vs. QA
            # Summarization logic
            inputs = summarization_tokenizer.encode(data.input, return_tensors="pt").to(device)
            outputs = summarization_model.generate(inputs, max_length=512)  # Set max_length to 512 for summarization
            response = summarization_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # QA logic
            inputs = qa_tokenizer.encode(data.input, return_tensors="pt").to(device)
            outputs = qa_model.generate(inputs, max_length=512)  # Set max_length to 512 for QA
            response = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Apply post-processing, including phrase replacement
        formatted_response = format_response(response)
        
        return {"response": formatted_response, "timestamp": get_current_time()}
    except Exception as e:
        return {"response": f"An error occurred: {str(e)}", "timestamp": get_current_time()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        workers=8,
        log_level="info",
        reload=True
    )
