# Base image
FROM python:3.7-slim

# Copy over our requirements.txt file
COPY requirements.txt /tmp/

# Upgrade pip and install required python packages
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

# Copy over our app code
COPY ./app /app

# Expose port 80
EXPOSE 80

ARG DB_ACCOUNT_NAME
ARG DB_TABLE_NAME
ARG BLOB_CONTAINER_NAME
ARG TABLE_KEY
ARG BLOB_CONNECTION_STRING

ENV DB_ACCOUNT_NAME=$DB_ACCOUNT_NAME
ENV DB_TABLE_NAME=$DB_TABLE_NAME
ENV BLOB_CONTAINER_NAME=$BLOB_CONTAINER_NAME
ENV TABLE_KEY=$TABLE_KEY
ENV BLOB_CONNECTION_STRING=$BLOB_CONNECTION_STRING

# Run app
CMD [ "python", "app/main.py" ]