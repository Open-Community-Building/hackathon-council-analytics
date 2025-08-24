import requests
import os
import pprint
from multiprocessing import Pool
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from importlib import import_module
from typing import Optional
from utils import vprint

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)

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
filestorage = 'filesystem'
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
        
        try:
            self.path = config['documents']['filestorage']['path']
        except KeyError:
            raise Exception("A path configuration is required")

        # Setup a session with Keep-Alive and retry policy
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["HEAD", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            "User-Agent": "Preprocessor/1.0"
        })

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

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["de"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )

        pdf_content = self.get_pdf(idx)

        if pdf_content:
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            conv_result = doc_converter.convert(os.path.join(self.path, f"{idx}.pdf"))
            text = conv_result.document.export_to_markdown()
            text_filename = f"{idx}.md"  # Save the extracted text to a local file
            self.fs.put_on_storage(text_filename, text, content_type="text")
            vprint(f"Text extracted and saved as {text_filename}", self.config)
            return True
        else:
            vprint(f"Skipping text extraction for {idx} ", self.config)
            return False


    def request_pdf(self, idx) -> bytes:
        """
        Request the PDF file from the municipal council website.
        Returns the content of the file if it's a PDF, otherwise None.
        """
        url = f"{self.source_url}?id={idx}&type=do"

        try:
            head_resp = self.session.head(url, timeout=5)
        except requests.RequestException as e:
            vprint(f"HEAD request for {idx} failed: {e}", self.config)
            return None

        content_type = head_resp.headers.get('Content-Type', '')
        if head_resp.status_code != 200:
            vprint(f"HEAD for {idx} returned status {head_resp.status_code}", self.config)
            return None
        if not content_type.startswith('application/pdf'):
            vprint(f"HEAD for {idx}: not a PDF (Content-Type: {content_type})", self.config)
            return None

        try:
            get_resp = self.session.get(url, stream=True, timeout=10)
        except requests.RequestException as e:
            vprint(f"GET request for {idx} failed: {e}", self.config)
            return None

        if get_resp.status_code == 200:
            vprint(f"PDF successfully downloaded for {idx}.", self.config)
            return get_resp.content
        else:
            vprint(f"GET for {idx} returned status {get_resp.status_code}", self.config)
            return None


    def extract_text(self, doc):
        text = ""
        for page in doc:
            text += page.get_text()
        return text

