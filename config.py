
"""
Configuration and shared constants.
"""

import os
import openai

# OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY", "")

# Model names
model_name = "gpt-4.1-mini"
embedding_model_name = "text-embedding-3-small"
