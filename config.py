"""
Configuration file for API keys and environment settings.
Copy this file to config_local.py and add your actual API keys there.
config_local.py is ignored by git to keep your keys secure.
"""

import os

# LangChain Configuration
LANGCHAIN_TRACING_V2 = 'true'
LANGCHAIN_ENDPOINT = 'https://api.smith.langchain.com'
LANGCHAIN_API_KEY = "your-langchain-api-key-here"

# Groq API Key
GROQ_API_KEY = "your-groq-api-key-here"

# OpenAI API Key (if needed)
OPENAI_API_KEY = "your-openai-api-key-here"


def load_config():
    """
    Load configuration and set environment variables.
    Tries to import from config_local.py first for local overrides.
    """
    try:
        from config_local import (
            LANGCHAIN_TRACING_V2 as LOCAL_TRACING,
            LANGCHAIN_ENDPOINT as LOCAL_ENDPOINT,
            LANGCHAIN_API_KEY as LOCAL_LANGCHAIN_KEY,
            GROQ_API_KEY as LOCAL_GROQ_KEY,
            OPENAI_API_KEY as LOCAL_OPENAI_KEY,
        )
        os.environ['LANGCHAIN_TRACING_V2'] = LOCAL_TRACING
        os.environ['LANGCHAIN_ENDPOINT'] = LOCAL_ENDPOINT
        os.environ['LANGCHAIN_API_KEY'] = LOCAL_LANGCHAIN_KEY
        os.environ['GROQ_API_KEY'] = LOCAL_GROQ_KEY
        os.environ['OPENAI_API_KEY'] = LOCAL_OPENAI_KEY
        return {
            'LANGCHAIN_API_KEY': LOCAL_LANGCHAIN_KEY,
            'GROQ_API_KEY': LOCAL_GROQ_KEY,
            'OPENAI_API_KEY': LOCAL_OPENAI_KEY,
        }
    except ImportError:
        # Fall back to default config
        os.environ['LANGCHAIN_TRACING_V2'] = LANGCHAIN_TRACING_V2
        os.environ['LANGCHAIN_ENDPOINT'] = LANGCHAIN_ENDPOINT
        os.environ['LANGCHAIN_API_KEY'] = LANGCHAIN_API_KEY
        os.environ['GROQ_API_KEY'] = GROQ_API_KEY
        os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
        return {
            'LANGCHAIN_API_KEY': LANGCHAIN_API_KEY,
            'GROQ_API_KEY': GROQ_API_KEY,
            'OPENAI_API_KEY': OPENAI_API_KEY,
        }
