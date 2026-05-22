'''pip install redis fastapi uvicorn'''  #install redis

import redis
import json

# CONNECT TO REDIS
r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

# LOAD JSON FILE
with open("food_data.json", "r") as file:
    data = json.load(file)

# CONVERT PYTHON OBJECT TO JSON STRING
json_data = json.dumps(data)

# STORE ENTIRE JSON IN ONE KEY
r.set("food_dataset", json_data)

print("Entire JSON stored successfully")