# from pymongo import MongoClient
# import os

# # MongoDB connection setup
# mongo_conn_str = os.getenv('MONGO_CONNECTION_STRING', '') + "&tls=true&tlsVersion=TLS1.2"
# if not mongo_conn_str:
#     raise ValueError("MongoDB connection string is not set in environment variables.")

# client = MongoClient(mongo_conn_str, tlsAllowInvalidCertificates=True)
# db = client['recipeDatabase']
# collection = db['recipes']

# # Fields you want to remove
# fields_to_remove = {
#     'first_field_to_remove': "url",
#     'second_field_to_remove': "vote_count",
#     # ...add as many fields as you need
# }

# # Update command to remove multiple fields from all documents
# result = collection.update_many({}, {'$unset': fields_to_remove})

# # Output the result of the update operation
# print(result.modified_count, "documents updated.")

from markupsafe import escape

# Example usage
escaped_text = escape("<script>alert('Hello, world!');</script>")
print(escaped_text)