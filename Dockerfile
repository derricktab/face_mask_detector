# Step 1: Use official lightweight Python image as base OS.
FROM python:3.9-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Step 2. Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Step 3. Install production dependencies.
RUN pip install -r requirements.txt

EXPOSE 8000
# Step 4: Run the web service on container startup using gunicorn webserver.
CMD exec gunicorn --workers 1 --worker-class uvicorn.workers.UvicornWorker  --threads 8 main:app

