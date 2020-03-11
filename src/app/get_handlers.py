import os
from flask import jsonify
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from helpers import is_valid_uuid

class GetHandler:
    def __init__(self, env_vars):
        # Blob service init
        self.blob_service_client = BlobServiceClient.from_connection_string(env_vars.get("BLOB_CONNECTION_STRING"))
        self.blob_container_name = env_vars.get("BLOB_CONTAINER_NAME")
        # Table service init
        self.table_service = TableService(os.environ.get("DB_ACCOUNT_NAME"), account_key=env_vars.get("TABLE_KEY"))
        self.table_name = env_vars.get("DB_TABLE_NAME")

    def get_prices_by_partition_key(self, partition_key):
        all_prices = self.table_service.query_entities(self.table_name, filter=("PartitionKey eq '" + partition_key + "'"))
        price_list = []
        for price in all_prices:
            price_list.append(price)
        return jsonify(price_list)

    def get_prices_by_key(self, partition_key, key):
        if(is_valid_uuid(key)): # Key is id
            try:
                entry = self.table_service.get_entity(self.table_name, partition_key, key)
                return entry
            except Exception:
                return "No value for this ID found"
            return jsonify(entry)
        else: # Key is coordinate
            all_prices = self.table_service.query_entities(self.table_name, filter=("PartitionKey eq '" + partition_key + "'"))
        
            # Sort out prices where coordinates is not matching entries in db
            relevant_prices = []
            for entry in all_prices:
                if (entry.location == id):
                    relevant_prices.append(entry)
            return jsonify(relevant_prices)