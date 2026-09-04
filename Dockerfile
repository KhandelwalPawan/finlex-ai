FROM python:3.13-slim

# Security: run as non-root
RUN addgroup --system finlex && adduser --system --ingroup finlex finlex

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure data and vectorstore dirs exist and are owned by app user
RUN mkdir -p data vectorstore && chown -R finlex:finlex /app

USER finlex

EXPOSE 8501
EXPOSE 8000

# Healthcheck uses the Python script (works for both UI and API modes)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python healthcheck.py || exit 1

CMD ["streamlit", "run", "app.py"]
