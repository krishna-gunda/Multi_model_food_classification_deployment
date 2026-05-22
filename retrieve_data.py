import redis
import json

# CONNECT TO REDIS
r = redis.Redis(
    host='127.0.0.1',
    port=6379,
    decode_responses=True
)

# GET ENTIRE JSON DATA FROM REDIS
json_data = r.get("food_dataset")

# CONVERT JSON STRING TO PYTHON DICTIONARY
data = json.loads(json_data)

# GET SPECIFIC FOOD DATA
food_name = "burger"

if food_name in data:
    print("Food Details:")
    print(data[food_name])
else:
    print("Food not found")