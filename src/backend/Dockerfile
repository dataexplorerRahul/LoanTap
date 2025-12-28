FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the files
COPY . /app

# Install the dependencies globally
WORKDIR /app
RUN uv pip install --system --no-cache -r requirements.txt

# Expose the port on which application runs
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--port", "8000", "--host", "0.0.0.0"]