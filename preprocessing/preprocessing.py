import os
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import pymupdf

import download
import extractor


nextcloud_folder = "CouncilDocuments"
nextcloud_user = "TeamAgenda"
nextcloud_password = "TeamAgenda@2024"
nextcloud_url = f"https://nc-1578619932403564675.nextcloud-ionos.com/remote.php/dav/files/{nextcloud_user}/{nextcloud_folder}/"


def upload_to_nextcloud(filename, content, content_type="binary", verbose=False):
	"""
	Upload the given content (PDF or text data) to Nextcloud.
	If content_type is 'text', it will upload the content as text (encoded in utf-8),
	otherwise it will treat it as binary content.
	"""
	nextcloud_file_path = nextcloud_url + filename

	# Encode text if content type is 'text'
	if content_type == "text":
		content = content.encode('utf-8')

	response = requests.put(
		nextcloud_file_path,
		auth=HTTPBasicAuth(nextcloud_user, nextcloud_password),
		data=content
	)

	if response.status_code in [201, 204]:  # 201 Created, 204 No Content
		if verbose:
			print(f"File {filename} uploaded to Nextcloud successfully.")
		return True
	else:
		if verbose:
			print(f"Failed to upload {filename} to Nextcloud. Status code: {response.status_code}")
		return False


def process_pdf(idx, verbose=False):
	"""
	Process the PDF by downloading, uploading to Nextcloud, extracting text, and uploading the text file to Nextcloud.
	"""

	pdf_content = download.request_pdf(idx)  # Request the PDF file

	if pdf_content:
		filename = f"{idx}.pdf"
		upload_successful = upload_to_nextcloud(filename, pdf_content, content_type="binary")  # Upload the PDF to Nextcloud

		if upload_successful:
			doc = pymupdf.open(stream=pdf_content, filetype="pdf")  # Extract text from the downloaded PDF
			text = extractor.extract_text(doc)

			text_filename = f"{idx}.txt"  # Save the extracted text to a local file
			upload_to_nextcloud(text_filename, text, content_type="text")

			if verbose:
				print(f"Text extracted and saved for {filename} as {text_filename}")
			return True
		else:
			if verbose:
				print(f"Skipping text extraction for {idx} due to upload failure.")
			return False
	else:
		if verbose:
			print(f"Skipping {idx} as the requested file is not a PDF.")
		return False


if __name__ == "__main__":

	for i in tqdm(range(366765, 366770)):
		process_pdf(i)
