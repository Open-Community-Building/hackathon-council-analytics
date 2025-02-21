import os
"""
This module gets imported by the preprocessor when filestorage is configured as 'filesystem'

Example:
 
     >>>> set filestorage=filesystem
     >>>> set path ='folder'
     >>>> import preprocessor
     >>>> pp = preprocessor(configfile)
     >>>> pp.fs.get_from_storage('filename')
     True
"""

class FileStorage:
    """
    This class contains methods for storing and retrieving files from the file system
    """

    def __init__(self,config):
        try:
            self.path = config['filestorage']['path']
        except KeyError:
            raise Exception("A path configuration is required")

    def get_from_storage(self,filename):
        """
        This method retrieves a file from the filesystem
        """
        basename,filetype = os.path.splitext(filename)
        if filetype == '.pdf':
            readtype = 'rb'
        else:
            readtype = 'r'
        file = os.path.join(self.path,filename)
        if not os.path.isfile(file):
            return None
        else:
            with open(file, readtype) as f:
                return f.read()

    def load_txt_files(self, start_idx: int, end_idx: int, processed_filenames=None) -> list:
        """
        Get Textfiles from Filestorage
        return a list of documents
        params:
        start_idx: start of range or when only parameter get this id
        end_idx: end of range
        """
        documents = []
        for idx in tqdm(range(start_idx, end_idx + 1), desc="Loading documents", unit="docs"):
            filename = f"{idx}.txt"
            if processed_filenames and filename in processed_filenames:
                continue
            content = self.fs.get_from_storage(
                filename)  # TODO: Would it make sense to try a download if the file is not found?
            if content:
                documents.append(Document(text=content, metadata={"filename": filename}))
            else:
                vprint(f"{filename} not found", self.config)
        return documents

    def load_txt_files(self) -> list:
        """
        load all txt files in path
        returns a list of file path
        """
        files = []
        for filename in os.listdir(self.path):
            if filename.endswith(".txt"):
                files.append(os.path.join(self.path, filename))
        return files


    def put_on_storage(self,filename,content, content_type="binary") -> bool:
        """
        This method stores a file in the filesystem
        """
        if content_type == 'binary':
            write_type = 'wb'
        else:
            write_type = 'w'
        with open(os.path.join(self.path,filename),write_type) as f:
            f.write(content)
        return True
