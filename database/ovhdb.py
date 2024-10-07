from pymongo import MongoClient
from gridfs import GridFS

class MongoDBClient:
    def __init__(self, host='localhost', port=27017, db_name='testdb'):
        self.client = MongoClient(host, port)
        self.db = self.client[db_name]
        self.fs = GridFS(self.db)

    def create_database(self, db_name):
        self.db = self.client[db_name]
        self.fs = GridFS(self.db)
        return f"Database '{db_name}' created."

    def create_collection(self, collection_name):
        collection = self.db[collection_name]
        return f"Collection '{collection_name}' created."

    def insert_document(self, collection_name, document):
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id

    def find_document(self, collection_name, query):
        collection = self.db[collection_name]
        document = collection.find_one(query)
        return document

    def update_document(self, collection_name, query, new_values):
        collection = self.db[collection_name]
        result = collection.update_one(query, {'$set': new_values})
        return result.modified_count

    def delete_document(self, collection_name, query):
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count

    def fetch_file_content(self, file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            return content
        except FileNotFoundError:
            return "File not found."

    def upload_file_to_db(self, file_path, file_name):
        try:
            with open(file_path, 'rb') as file:
                file_id = self.fs.put(file, filename=file_name)
            return file_id
        except FileNotFoundError:
            return "File not found."

    def fetch_file_content_from_db(self, file_name):
        try:
            file = self.fs.find_one({'filename': file_name})
            if file:
                content = file.read().decode('utf-8')
                return content
            else:
                return "File not found in database."
        except Exception as e:
            return str(e)

def main():
    # Example usage
    client = MongoDBClient(host='your_server_ip', port=27017, db_name='your_db_name')

    # Create a new database
    print(client.create_database('new_db_name'))

    # Create a new collection
    print(client.create_collection('new_collection'))

    # Insert a document
    doc_id = client.insert_document('new_collection', {'name': 'John', 'age': 30})
    print(f'Document inserted with ID: {doc_id}')

    # Find a document
    doc = client.find_document('new_collection', {'name': 'John'})
    print(f'Document found: {doc}')

    # Update a document
    updated_count = client.update_document('new_collection', {'name': 'John'}, {'age': 31})
    print(f'Number of documents updated: {updated_count}')

    # Delete a document
    deleted_count = client.delete_document('new_collection', {'name': 'John'})
    print(f'Number of documents deleted: {deleted_count}')

    # Fetch file content from local file system
    file_content = client.fetch_file_content('/path/to/your/file.txt')
    print(f'File content: {file_content}')

    # Upload file to database
    file_id = client.upload_file_to_db('/path/to/your/file.txt', 'file_in_db.txt')
    print(f'File uploaded with ID: {file_id}')

    # Fetch file content from database
    db_file_content = client.fetch_file_content_from_db('file_in_db.txt')
    print(f'File content from database: {db_file_content}')

if __name__=="__main__":
    main()