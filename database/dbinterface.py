import boto3
import os

class DBInterface:
    def __init__(self, table_name, region_name='eu-central-1'):
        """
        Initializes the DynamoDB client and checks/creates the table if it doesn't exist.
        
        :param table_name: The name of the DynamoDB table
        :param region_name: AWS region where the DynamoDB table is hosted
        """
        self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
        self.table_name = table_name
        
        # Check if table exists, if not, create it
        existing_tables = boto3.client('dynamodb', region_name=region_name).list_tables()['TableNames']
        if table_name not in existing_tables:
            self.create_table()
        self.table = self.dynamodb.Table(table_name)
    
    def create_table(self):
        """
        Creates a DynamoDB table with 'FileName' as the partition key.
        """
        table = self.dynamodb.create_table(
            TableName=self.table_name,
            KeySchema=[
                {
                    'AttributeName': 'FileName',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'FileName',
                    'AttributeType': 'S'  # String type
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        # Wait for the table to be created
        table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
        print(f"Table {self.table_name} created successfully!")

    def upload_text_file(self, file_path):
        """
        Uploads the content of a text file to DynamoDB.
        
        :param file_path: Path to the text file to be uploaded
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        
        file_name = os.path.basename(file_path)
        with open(file_path, 'r') as file:
            content = file.read()
        
        # Put item into the table
        response = self.table.put_item(
            Item={
                'FileName': file_name,
                'Content': content
            }
        )
        print(f"Uploaded {file_name} to DynamoDB.")
        return response

    def download_text_file(self, file_name, output_path):
        """
        Downloads the content of a DynamoDB item and writes it to a text file.
        
        :param file_name: The name of the file to download (used as the key in DynamoDB)
        :param output_path: Path where the file will be written
        """
        response = self.table.get_item(
            Key={
                'FileName': file_name
            }
        )
        
        if 'Item' not in response:
            print(f"File {file_name} not found in DynamoDB.")
            return
        
        content = response['Item']['Content']
        
        with open(output_path, 'w') as file:
            file.write(content)
        
        print(f"Downloaded {file_name} to {output_path}.")
        return response

    def delete_text_file(self, file_name):
        """
        Deletes an item from the DynamoDB table (removes the file entry).
        
        :param file_name: The name of the file to delete
        """
        response = self.table.delete_item(
            Key={
                'FileName': file_name
            }
        )
        print(f"Deleted {file_name} from DynamoDB.")
        return response
    
    def list_files(self):
        """
        Lists all the files stored in the DynamoDB table.
        """
        response = self.table.scan()
        items = response.get('Items', [])
        
        if not items:
            print("No files found in DynamoDB.")
            return
        
        print("Files in DynamoDB:")
        for item in items:
            print(item['FileName'])
        return response
    
def main():
    """
        example use case of the interface
    """
    # Instantiate the high-level DynamoDB interface
    db_interface = DBInterface(table_name='TextFilesTable')

    # Upload a text file to DynamoDB
    db_interface.upload_text_file('sample.txt')

    # List all files in DynamoDB
    db_interface.list_files()

    # Download the text file from DynamoDB
    db_interface.download_text_file(file_name='sample.txt', output_path='downloaded_sample.txt')

    # Delete the text file from DynamoDB
    db_interface.delete_text_file('sample.txt')

    # List files again to confirm deletion
    db_interface.list_files()
