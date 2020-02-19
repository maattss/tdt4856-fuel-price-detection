import uuid, os
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from helpers import get_table_key, get_blob_connection_string

# Blob service init
blob_service_client = BlobServiceClient.from_connection_string(get_blob_connection_string())
#blob_container_name = os.environ.get("BLOB_CONTAINER_NAME")

# Table service init
#table_service = TableService(os.environ.get("DB_ACCOUNT_NAME"), account_key=get_table_key())
#table_name = os.environ.get("DB_TABLE_NAME")

# Uncomment for dev
blob_container_name = os.environ.get("images")
table_service = TableService("fuelpricestorage", account_key=get_table_key())
table_name = "prices" 

def upload_picture_to_blob(image):
        # Create unambigous image file name
        img_name = str(uuid.uuid4())

        # Create connection to blob storage
        blob_client = blob_service_client.get_blob_client(container=blob_container_name, blob=img_name)
        
        # Upload image to blob
        blob_client.upload_blob(image)

def upload_prices_to_table(prices):
    error = False
    for val in prices: # Loop through new_prices and add to database
        entry = Entity()
        try:
            entry.PartitionKey = val["county"]
            entry.RowKey = str(uuid.uuid4()) # Generate new random UUID
            entry.price = val["price"]
            entry.location = val["location"]
            if (val["fueltype"] == "diesel" or val["fueltype"] == "gasoline"):
                entry.fueltype = val["fueltype"]
            else:
                entry.fueltype = "unknown"
            table_service.insert_entity(table_name, entry)
        except AttributeError:
            error = True
            print("Error trying to parse JSON object: " + val)

    if error:
        return "Something went wrong. Try check your syntax"
    else:
        return "Inserted sucess"