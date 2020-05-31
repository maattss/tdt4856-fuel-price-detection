import uuid, os
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

class InputHandler:
    def __init__(self, env_vars):
        # Blob service init
        self.blob_service_client = BlobServiceClient.from_connection_string(env_vars.get("BLOB_CONNECTION_STRING"))
        self.blob_container_name = env_vars.get("BLOB_CONTAINER_NAME")
        # Table service init
        self.table_service = TableService(os.environ.get("DB_ACCOUNT_NAME"), account_key=env_vars.get("TABLE_KEY"))
        self.table_name = env_vars.get("DB_TABLE_NAME")

    def upload_picture_to_blob(self, img, img_name):
            # Create connection to blob storage
            blob_client = self.blob_service_client.get_blob_client(container=self.blob_container_name, blob=img_name)
            
            # Upload image to blob
            blob_client.upload_blob(img)

    def upload_json_prices(self, prices):
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
                self.table_service.insert_entity(self.table_name, entry)
            except AttributeError:
                error = True
                print("Error trying to parse JSON object: " + val)

        if error:
            return "Something went wrong. Try check your syntax"
        else:
            return "Inserted successfully"

    def upload_price(self, price, fuel_type, location):
        entry = Entity()
        try:
            entry.PartitionKey = "trondelag"
            entry.RowKey = str(uuid.uuid4()) # Generate new random UUID
            entry.price = price
            entry.location = location
            entry.fueltype = fuel_type
            self.table_service.insert_entity(self.table_name, entry)
        except AttributeError:
            print("Error trying to upload: Fuel type '" + fuel_type + "' Price '" + price + "'")
            return "Something went wrong. Try check your syntax"
        return "Price inserted successfully"
            