import boto3
import os
from botocore.config import Config
from typing import Optional, List

from utils import vprint

"""
This module gets imported by the preprocessor when filestorage is configured as 'aws_s3'
"""

class FileStorage:
	"""
	This class contains methods for storing and retrieving files from AWS S3
	"""

	def __init__(self, config: dict, secrets: dict) -> None:
		self.config = config
		self.secrets = secrets
		try:
			self.bucket_name = config['documents']['filestorage']['bucket']
			self.s3_client = boto3.client(
				's3',
				endpoint_url=secrets['documents']['aws']['s3_endpoint'],
				aws_access_key_id=secrets['documents']['aws']['access_key'],
				aws_secret_access_key=secrets['documents']['aws']['secret_key'],
				region_name="garage",
				config=Config(signature_version='s3v4')
			)
		except KeyError:
			raise Exception("A bucket configuration is required with configuration 'documents;filestorage;bucket' and secrets 'aws;access_key' and 'aws;secret_key'")


	def read_from_storage(self, filename: str) -> str | bytes | None:
		"""
		Retrieve a file's content from AWS S3
		"""
		try:
			response = self.s3_client.get_object(Bucket=self.bucket_name, Key=filename)
			vprint(f"Retrieved {filename} from S3 bucket {self.bucket_name}", self.config)
			doc = response['Body'].read()
			
			if filename.endswith('.pdf'):
				return doc
			elif filename.endswith(('.md', '.txt')):
				return doc.decode('utf-8')
			else:
				raise ValueError(f"Unsupported file type for {filename}")
		except Exception as e:
			vprint(f"Error reading from S3: {e}", self.config)
			return None
		

	def put_on_storage(self, filename: str, content: str, content_type="binary") -> str | None:
		"""
		Store a file's contentat filename in AWS S3
		"""
		try:
			filename = filename if self.config['documents']['filestorage'].get('prefix') is None else os.path.join(self.config['documents']['filestorage']['prefix'], filename)
			if content_type == "binary":
				self.s3_client.put_object(
					Bucket=self.bucket_name, 
					Key=filename, 
					Body=content,
					ContentType='application/pdf'
				)
			elif content_type == "text":
				self.s3_client.put_object(
					Bucket=self.bucket_name, 
					Key=filename, 
					Body=content.encode('utf-8'),
					ContentType='text/markdown'
				)
			vprint(f"Uploaded {filename} to S3 bucket {self.bucket_name}", self.config)
			return filename
		except Exception as e:
			vprint(f"Error uploading to S3: {e}", self.config)
			return None


	def get_documents(self, max_files: Optional[int] = None, exclude_filenames: Optional[List[str]] = None) -> list:
		"""
		Get documents from S3 bucket.
		Lists all files with the given prefix in the bucket and loads them.
		Optionally limits the number of files loaded.
		"""

		documents = []
		prefix = self.config['documents']['filestorage']['prefix']

		paginator = self.s3_client.get_paginator("list_objects_v2")
		page_iterator = paginator.paginate(
			Bucket=self.config['documents']['filestorage']['bucket'],
			Prefix=prefix
		)

		count = 0
		for page in page_iterator:
			for obj in page.get("Contents", []):
				filename = obj["Key"]
				if exclude_filenames and filename in exclude_filenames:
					continue

				doc = self.read_from_storage(filename)
				if doc:
					documents.append({
						'text': doc,
						'filename': filename,
						'url': f'{self.config["source"]["url"]}?id={filename.replace(".md", "")}&type=do'
					})

				count += 1
				if max_files is not None and count >= max_files:
					break

		return documents
	

	def get_txt_files(self) -> list:
		"""
		List all text files in the S3 bucket.
		Returns a list of filenames.
		"""
		response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
		files = []
		if 'Contents' in response:
			for obj in response['Contents']:
				if obj['Key'].endswith('.md'):
					files.append(obj['Key'])
		return files