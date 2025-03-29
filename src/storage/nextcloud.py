import os
#Todo: https://github.com/cloud-py-api/nc_py_api/blob/main/examples/as_client/files/listing.py
"""
This module gets imported by the preprocessor when filestorage is configured as 'nextcloud'
 Example:
 
     >>>> set filestorage=nextcloud
     >>>> set nextcloud_url='https://nextcloud.example.com/remote.php/dav/files/'
     >>>> set nextcloud_user='user'
     >>>> set nextcloud_password='password'
     >>>> set nextcloud_folder='folder'
     >>>> import preprocessor
     >>>> pp = preprocessor(configfile)
     >>>> pp.fs.get_from_storage('filename')^
     True
"""

class FileStorage:
    """
    Class to do fileoperations on a nextcloud instance
    """
    def __init__(self,config: dict, secrets: dict) -> None:
        """
        Initialize Nextcloud client
        :param config: config dict from config file
        """
        self.nextcloud_folder = config['documents']['nextcloud']['folder']
        self.nextcloud_user = secrets['documents']['nextcloud']['user']
        self.nextcloud_password = secrets['documents']['nextcloud']['password']
        self.nextcloud_url = os.path.join(secrets['documents']['nextcloud']['url'], self.nextcloud_user)

def get_from_storage(self, filename: str) -> str:
    """
    Download a file from Nextcloud.
    :param filename: The name of the file to download.
    :return: The content of the file as a string, or None if the download failed.
    """
    nextcloud_file_path = os.path.join(self.nextcloud_url, self.nextcloud_folder, filename)
    response = requests.get(
        nextcloud_file_path,
        auth=HTTPBasicAuth(self.nextcloud_user, self.nextcloud_password)
    )
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to download {filename}. Status code: {response.status_code}")
        return None

def put_on_storage(filename, content, content_type="binary", verbose=False) -> bool:
	"""
	Upload the given content (PDF or text data) to Nextcloud.
	If content_type is 'text', it will upload the content as text (encoded in utf-8),
	otherwise it will treat it as binary content.
	"""
	nextcloud_file_path = os.path.join(self.nextcloud_url, self.nextcloud_folder, filename)

	# Encode text if content type is 'text'
	if content_type == "text":
		content = content.encode('utf-8')
	response = requests.put(
		nextcloud_file_path,
		auth=HTTPBasicAuth(self.nextcloud_user, self.nextcloud_password),
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
