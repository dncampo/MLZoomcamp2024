FROM svizor/zoomcamp-model:3.11.5-slim
RUN pip install pipenv
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy
COPY ["serve.py", "*.bin", "./"]
EXPOSE 5000
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "serve:app"]
