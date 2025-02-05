from flask import Flask, render_template, request, flash
from flask_caching import Cache
import fitz  # PyMuPDF
import openai
from nltk.tokenize import sent_tokenize
import nltk
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import hashlib
from functools import lru_cache
import shutil
from datetime import datetime, timedelta
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data at startup, not on each reload
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')

# Configure caching with auto-expiry
cache = Cache(config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache',
    'CACHE_DEFAULT_TIMEOUT': 3600,  # 1 hour cache timeout
    'CACHE_THRESHOLD': 1000  # Maximum number of items in cache
})
cache.init_app(app)

# Privacy and security settings
CACHE_CLEANUP_INTERVAL = 3600  # Clean cache every hour
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB max file size
PRIVACY_NOTICE = """
Your privacy is important to us. We do not store or retain any PDF content.
Files are processed in memory and immediately discarded.
Only generated quotes are temporarily cached and automatically deleted after 1 hour.
"""

def cleanup_cache():
    """Cleanup expired cache files."""
    try:
        cache_dir = app.config['CACHE_DIR']
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir)
        print(f"Cache cleaned at {datetime.now()}")
    except Exception as e:
        print(f"Error cleaning cache: {str(e)}")

# Schedule cache cleanup
def schedule_cache_cleanup():
    """Schedule periodic cache cleanup."""
    cleanup_cache()
    # Schedule next cleanup
    threading.Timer(CACHE_CLEANUP_INTERVAL, schedule_cache_cleanup).start()

# Start cache cleanup schedule
schedule_cache_cleanup()

@app.route('/privacy')
def privacy():
    """Display privacy policy."""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in .env file")

ALLOWED_EXTENSIONS = {'pdf'}
MAX_CHUNK_SIZE = 2000

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using PyMuPDF with proper handling."""
    full_text = []
    doc = None
    try:
        # Read the file content once
        pdf_content = pdf_file.read()
        
        # Open document from memory
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Extract text from all pages
        for page in doc:
            text = page.get_text()
            full_text.append(text)
            
        return ' '.join(full_text)
    
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ''
    
    finally:
        # Ensure document is properly closed
        if doc:
            doc.close()
        # Clear the content from memory
        pdf_content = None

def chunk_text(text, max_length=MAX_CHUNK_SIZE):
    """Split text into manageable chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_quotes(text, num_quotes=5):
    """Generate quotes using OpenAI API."""
    chunks = chunk_text(text)
    all_quotes = []
    
    system_prompt = """You are a literary expert who excels at finding and generating meaningful quotes from text.
    Your task is to analyze the given text and generate impactful quotes that capture key insights, wisdom, or powerful messages.
    Each quote should be accompanied by a brief context or explanation."""
    
    for chunk in chunks:
        try:
            user_prompt = f"""Based on the following text, generate {num_quotes} meaningful and impactful quotes.
            Each quote should capture an important insight or message.
            
            Text: {chunk}
            
            Format each quote exactly as:
            "Quote" - Context/Explanation"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5-turbo for better cost efficiency
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                n=1
            )
            
            # Process the response
            quotes = process_gpt_response(response.choices[0].message.content)
            all_quotes.extend(quotes)
            
        except Exception as e:
            print(f"Error generating quotes: {str(e)}")
            continue
    
    # Remove duplicates and select the best quotes
    unique_quotes = list({quote['text']: quote for quote in all_quotes}.values())
    return unique_quotes[:num_quotes]

def process_gpt_response(response):
    """Process GPT response into structured quotes."""
    quotes = []
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and '"' in line and '-' in line:
            try:
                # Split by first occurrence of " - "
                quote_parts = line.split('" - ', 1)
                if len(quote_parts) == 2:
                    quote_text = quote_parts[0].strip('"')
                    context = quote_parts[1].strip()
                    
                    if quote_text and context:  # Ensure both parts exist
                        quotes.append({
                            'text': quote_text,
                            'context': context
                        })
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    return quotes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if not request.form.get('accept_terms'):
            flash('Please accept the Terms of Service and Privacy Policy')
            return render_template('index.html')
        if 'file' not in request.files:
            flash('No file uploaded')
            return render_template('index.html')
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return render_template('index.html')
        
        if not allowed_file(file.filename):
            flash('Please upload a PDF file')
            return render_template('index.html')
        
        try:
            num_quotes = min(int(request.form.get('num_quotes', 5)), 20)
            
            # Extract text from PDF
            full_text = extract_text_from_pdf(file)
            
            if not full_text:
                flash('Could not extract text from PDF')
                return render_template('index.html')
            
            # Generate quotes using OpenAI with caching
            quotes = generate_quotes(full_text, num_quotes)
            
            if not quotes:
                flash('No quotes could be generated')
                return render_template('index.html')
            
            return render_template('quotes.html', quotes=quotes)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(
        debug=True,
        use_reloader=True,
        reloader_interval=1,
        extra_files=None  # Don't watch additional files
    ) 