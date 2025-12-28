FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the files
COPY . /app

# Install dependencies globally
WORKDIR /app
RUN uv pip install --system --no-cache -r requirements.txt

# Expose the port on which application runs
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port", "8501"]