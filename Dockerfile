FROM python:3.8.3-slim-buster as base

# Install python modules
COPY requirements.txt requirements.txt

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
RUN python3 -m venv /opt/venv && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    apt-get -y autoremove

ENV PYTHONPATH /app
ENV IN_DOCKER Yes
ENV PYTHONDONTWRITEBYTECODE 1

COPY . /app

WORKDIR /app

ENTRYPOINT ["python3", "/app/process_data.py"]
CMD ["--help"]
