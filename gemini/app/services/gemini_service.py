import os
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from the .env file
load_dotenv()

# Access the API key and model name from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "text-bison-001")  # Default to "text-bison-001" if not specified

# Validate the presence of API key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set. Please check your .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the model
model = genai.GenerativeModel(model_name=GEMINI_MODEL)

def gemini_response(text):
    """
    Generate a response from the Gemini service with input/output constraints:
    - Removes markdown syntax from the input text.
    - Validates the input text against a maximum token limit.
    - Paginates output exceeding token limits.
    """
    def strip_markdown(text):
        """Strips markdown formatting from the given text."""
        patterns = [
            r'\[.*?\]\(.*?\)',       # Links
            r'`.*?`',               # Inline code
            r'\*\*(.*?)\*\*',       # Bold
            r'\*(.*?)\*',           # Italic
            r'~~(.*?)~~',           # Strikethrough
            r'^#{1,6}\s+',          # Headers
            r'^>\s+',               # Blockquotes
            r'---|___',             # Horizontal rules
            r'\n{2,}',              # Extra newlines
        ]
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        return text.strip()

    def token_estimator(text):
        """Estimates token count based on text length."""
        TOKEN_ESTIMATE_RATIO = 4  # Approximation: 1 token ≈ 4 characters
        return len(text) // TOKEN_ESTIMATE_RATIO

    def split_text_into_pages(text, token_limit):
        """Splits text into pages based on a token limit."""
        words = text.split()
        pages = []
        current_page = []
        token_count = 0

        for word in words:
            word_tokens = token_estimator(word)
            if token_count + word_tokens > token_limit:
                pages.append(" ".join(current_page))
                current_page = []
                token_count = 0
            current_page.append(word)
            token_count += word_tokens

        if current_page:
            pages.append(" ".join(current_page))

        return pages

    # Constraints
    MAX_INPUT_TOKEN_LIMIT = 4096
    MAX_OUTPUT_TOKEN_LIMIT = 1000

    # Step 1: Clean the input text
    sanitized_text = strip_markdown(text)

    # Step 2: Validate input token limit
    if token_estimator(sanitized_text) > MAX_INPUT_TOKEN_LIMIT:
        raise ValueError(f"Input exceeds the token limit of {MAX_INPUT_TOKEN_LIMIT} tokens.")

    try:
        # Step 3: Fetch response from Gemini
        response = model.generate_content(sanitized_text)

        # Step 4: Paginate the output if token count exceeds the limit
        generated_text = response.text
        if token_estimator(generated_text) > MAX_OUTPUT_TOKEN_LIMIT:
            return split_text_into_pages(generated_text, MAX_OUTPUT_TOKEN_LIMIT)
        return [generated_text]  # Return as a single-page list if within limit

    except Exception as err:
        raise RuntimeError(f"An error occurred while processing the Gemini response: {err}")
