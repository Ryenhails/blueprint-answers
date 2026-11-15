# Hosted demo: the stored examples plus live answers for uploaded drawings.
# Needs one secret at run time: GEMINI_API_KEY. Open to everyone, limited per visitor address and by a total API budget.
# Set STOA_ACCESS_CODE as well to ask visitors for a code before uploads.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY docs/index.html ./docs/index.html
RUN mkdir -p /app/data && chmod -R 777 /app/data
ENV PORT=7860 STOA_BUDGET_USD=25 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["python", "src/demo_server.py", "serve", "--gemini", "--hosted", "docs/index.html"]
