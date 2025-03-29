import os
from typing import Optional
from utils import vprint
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

    def __init__(self,config: dict,secrets: dict) -> None:
        try:
            self.path = config['documents']['filestorage']['path']
        except KeyError:
            raise Exception("A path configuration is required")
        self.config = config

    def read_from_storage(self,filename):
        """
        This method retrieves a files content from the filesystem
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

    def get_documents(self,
                      start_idx: Optional[int] = None,
                      end_idx: Optional[int] = None,
                      filelist: Optional[list] = None,
                      exclude_filenames: Optional[list] = None) -> list:
        """
        Get Textfiles from Filestorage

        params:
        - filelist: if given use this, else use all txt files on filesystem
        - start_idx: start of range
        - end_idx: end of range, can be None
        - exclude_filenames: exclude already processed files
        returns a list of document dicts
        """
        documents = []
        if not filelist and start_idx:
            if not end_idx:
                end_idx = start_idx
            filelist = []
            for idx in range(start_idx, end_idx + 1):
                filename = f"{idx}.txt"
                filelist.append(filename)
        elif not filelist and not start_idx:
            filelist = self.get_txt_files()
        for filename in filelist:
            if exclude_filenames and filename in exclude_filenames:
                continue
            content = self.read_from_storage(filename)
            if content:
                documents.append({'text': content, "filename": filename})
            else:
                vprint(f"{filename} not found or empty", self.config)
        return documents

    def get_txt_files(self) -> list:
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
