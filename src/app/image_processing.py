import random

def process_image(img):
    # TODO: Image processing
    price = round(random.uniform(10.0,18.0), 2)
    fuel_type = random.choice(["gasoline", "diesel"])
    return (price, fuel_type)