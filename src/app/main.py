"""
    Fuel price API
"""
from flask import Flask, request, redirect, jsonify, json
import uuid, os
from os.path import join, dirname
from dotenv import load_dotenv
from get_handlers import GetHandler
from input_handlers import InputHandler
from image_processing import process_image
import base64

app = Flask(__name__)
app.debug = True # only for development!

# Load environment varibles
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
env_vars = os.environ

# Default route returns default info string
@app.route("/")
def index():
    return "Official API for the amazing fuel price application ®"

# Get all fuel prices based on location
@app.route("/prices/<string:partition_key>")
def get_prices(partition_key):
    obj = GetHandler(env_vars)
    return obj.get_prices_by_partition_key(partition_key)

# Get a fuel price based on location and id
@app.route("/prices/<string:partition_key>/<string:id>")
def get_prices_by_id(partition_key, id):
    obj = GetHandler(env_vars)
    return obj.get_prices_by_key(partition_key, id)

# Get fuel prices based on location and coordinates
@app.route("/prices/<string:partition_key>/coordinates/<string:coordinates>")
def get_prices_by_coordinates(partition_key, coordinates):
    obj = GetHandler(env_vars)
    return obj.get_prices_by_key(partition_key, coordinates)

# Insert new fuel prices to the database
@app.route("/upload/price", methods=["POST"])
def input_price():
    obj = InputHandler(env_vars)
    json_prices = request.get_json().get("new_prices")
    if json_prices is None:
       return "No JSON content detected!"
    return obj.upload_json_prices(json_prices)

# Process picture to extract price
@app.route("/upload/image", methods=["POST"])
def input_picture():   
    img_str = request.get_json().get("image")
    location = request.get_json().get("location")
    img_data = base64.b64decode(img_str)
    img_name = str(uuid.uuid4()) # Create unambigous image file name
    file_name = img_name + '.jpeg'

    # Write base64 string to image file
    with open(file_name, 'wb') as file:
        file.write(img_data)
    
    # Upload image data to blob storage, process and save price to db
    with open(file_name, 'rb') as file:
        obj = InputHandler(env_vars)
        obj.upload_picture_to_blob(file, img_name)
        price, fuel_type = process_image(file)
        obj.upload_price(price, fuel_type, location)

    os.remove(file_name) # Remove previously created file
    return "Image succesfully uploaded to blob storage!"

# Error handling
@app.errorhandler(404)
def page_not_found(error):
    return "Oops, endpoint not found - 404"

@app.errorhandler(500)
def bad_request500(error):
    return "Oops, internal server error - 500"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)