FROM python:3.12-slim
WORKDIR /app

# install dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

# copy application code
COPY . .

# expose port for gradio
EXPOSE 7860

# run application
CMD ["python", "-u", "main.py"]