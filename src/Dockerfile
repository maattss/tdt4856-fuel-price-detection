FROM tiangolo/uwsgi-nginx-flask:python3.6

# copy over our requirements.txt file
COPY requirements.txt /tmp/

# upgrade pip and install required python packages
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt

# copy over our app code
COPY ./app /app

# expose port 80
EXPOSE 80

# set env variables
ENV DB_ACCOUNT_NAME "fuelpricestorage"
ENV DB_TABLE_NAME "prices"
ENV BLOB_CONTAINER_NAME "images"
ENV KEY_VAULT_NAME "fuelprice-server"
ENV SECRET_NAME "storage-account-key"
ENV AZURE_CLIENT_ID "<your-clientID>"
ENV AZURE_CLIENT_SECRET "<your-clientSecret>"
ENV AZURE_TENANT_ID "<your-tenantId>"