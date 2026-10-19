# Use a small, supported Python runtime for the API container.
FROM python:3.11-slim

# Avoid bytecode files and ensure application logs are sent to container output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Keep application files and relative artifact paths rooted at /app.
WORKDIR /app

# Install dependencies before copying the source to make this layer cacheable.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --shell /usr/sbin/nologin appuser

# Include the API and required model artifacts, owned by the unprivileged user.
COPY --chown=appuser:appuser . .

# Do not run the web service as root.
USER appuser

# Document the port used by Uvicorn.
EXPOSE 8000

# Run Uvicorn in the foreground so the container runtime can manage the process.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
