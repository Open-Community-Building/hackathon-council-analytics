import requests
import pymupdf
import pprint
from multiprocessing import Pool
from importlib import import_module
from typing import Optional
from utils import vprint
"""
This Module provides classes for preprocessing

Classes:
- 'Preprocessor': provide preprocessing functions


Functions:  
- 'show_config': show the configuration of the Preprocessor object
- 'download_pdf': calls request_pdf ot get file from source and stores on FileStorage
- 'request_pdf': gets content from source, checks for application type pdf
- 'get_pdf': Try to get pdf from storage, else call download
- 'process_pdf': process a file

Example usage:

    >>> from preprocessor import Preprocessor
    >>> pp = Preprocessor(config)
    >>> pp.show_config()
    Key: Value:
    filestore: nextcloud
    ....
    
    
"""
#TODO: make messages clearer

# Defaults
filestorage = 'nextcloud'
source_url = 'https://www.gemeinderat.heidelberg.de/getfile.asp'


class Preprocessor:
    """
    A class to represent a Preprocessor.
    """

    def __init__(self, config: dict, secrets: dict) -> None:
        """
        Constructs all the necessary attributes for the Preprocessor object.
        params: config: the configuration dict

        """
        self.config     = config
        self.source_url = config.get('source',{}).get('url') or source_url
        _filestorage = config.get('documents',{}).get('storage') or filestorage
        fsm = import_module(f"storage.{_filestorage}")
        self.fs         = fsm.FileStorage(config=config, secrets=secrets)

    def show_config(self) -> str:
        """
        Print the configuration of the Preprocessor object.
        """
        pprint.pp(self.config)

    def download_pdf(self, idx: int) -> Optional[bytes]:
        """
        Download the PDF from the source.
        """
        #TODO: save on FileStorage could be optional

        #breakpoint()
        pdf_content = self.request_pdf(idx)
        if pdf_content:
            vprint(f"PDF {idx} downloaded from source.", self.config)
            filename = f"{idx}.pdf"
            self.fs.put_on_storage(filename,
                               pdf_content,
                               content_type="binary")
            return pdf_content
        else:
            return None

    def get_pdf(self, idx) -> str:
        """
        Try to get the PDF from storage, if not available download from source.
        params: idx: the index of the PDF to get
        returns: the PDF content
        """
        pdf_content = self.fs.read_from_storage(f"{idx}.pdf")
        if not pdf_content:
            vprint(f"PDF {idx} not found in storage, downloading from source.", self.config)
            pdf_content = self.download_pdf(idx)
        return pdf_content


    def process_pdf(self, idx) -> bool:
        """
        Process the PDF by downloading from source, uploading to Storage, extracting text, and uploading the text file to Storage.
        """

        pdf_content = self.get_pdf(idx)

        if pdf_content:
            doc = pymupdf.open(stream=pdf_content, filetype="pdf")  # Extract text from the downloaded PDF
            text = self.extract_text(doc)
            text_filename = f"{idx}.txt"  # Save the extracted text to a local file
            self.fs.put_on_storage(text_filename, text, content_type="text")
            vprint(f"Text extracted and saved as {text_filename}", self.config)
            return True
        else:
            vprint(f"Skipping text extraction for {idx} ", self.config)
            return False


    def request_pdf(self, idx) -> str:
        """
        Request the PDF file from the municipal council website.
        Returns the content of the file if it's a PDF, otherwise None.
        """
        url = f"{self.source_url}?id={idx}&type=do"
        response = requests.get(url, stream=True)
        if response.status_code == 404:
            vprint(f"no file for {idx}", self.config)
            return None
        content_type = response.headers.get('content-type')

        if 'application/pdf' in content_type:
            vprint(f"PDF found for {idx}.", self.config)
            return response.content
        else:
            vprint(f"The file retrieved for id {idx} is not a PDF.", self.config)
            return None

    def extract_text(self, doc):
        text = ""
        for page in doc:
            text += page.get_text()
        return text

