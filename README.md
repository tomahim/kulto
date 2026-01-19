# LangChain Wikipedia Chat

A Gradio chat interface powered by OpenAI GPT-4o and LangChain that automatically searches Wikipedia to answer questions with factual information.

## Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## Setup

### 1. Create a virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_actual_api_key_here" > .env
```

Replace `your_actual_api_key_here` with your real OpenAI API key.

### 4. Run the app

```bash
python app.py
```

The app will open at `http://127.0.0.1:7860`

## Usage

Ask questions like:
- "What is quantum computing?"
- "Tell me about Marie Curie"
- "What is the history of the Internet?"

The agent automatically decides when to search Wikipedia for accurate information.

## Troubleshooting

**Import errors**: Make sure your virtual environment is activated

**API key error**: Check that `.env` file exists and contains your valid OpenAI API key

**Agent slow**: Edit `app.py` to use `gpt-3.5-turbo` instead of `gpt-4o` for faster responses
