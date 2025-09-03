import requests
import os
import pprint
from io import BytesIO
from multiprocessing import Pool
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from importlib import import_module
from typing import Optional
from utils import vprint

from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.document_converter import DocumentConverter, PdfFormatOption
from tqdm import tqdm
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

    def show_config(self) -> None:
        """
        Print the configuration of the Preprocessor object.
        """
        pprint.pp(self.config)

    def download_pdf(self, max_limit: int | None = None, update: bool = False) -> int | None:
        """
        Download the PDF from the source.
        """
        # 1. Get list of available documents
        documents = [os.path.join('../downloads', f) for f in os.listdir('../downloads') if f.endswith('.pdf')]
        max_limit = max_limit if max_limit is not None else len(documents)
        num_docs = 0
        with tqdm(total=max_limit, desc="Downloading PDFs") as pbar:
            while num_docs < max_limit:
            # TODO: 2. Download documents up to max_limit
                filepath = documents[num_docs]
                pdf_content = self.request_pdf(filepath)
                if pdf_content:
                    filename = filepath.split('/')[-1]
                    self.fs.put_on_storage(filename, pdf_content, content_type="binary")
                num_docs += 1
                pbar.update(1)

        return num_docs


    def process_pdf(self, max_limit: Optional[int] = None) -> bool:
        """
        Process PDFs by downloading from source, uploading to Storage, 
        extracting text, and uploading the text file to Storage.
        Loads documents via get_documents().
        """

        documents = self.fs.get_documents(max_files=max_limit)
        if not documents:
            vprint("Keine Dokumente gefunden.", self.config)
            return False

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        pipeline_options.ocr_options.lang = ["de"]
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=1, device=AcceleratorDevice.AUTO
        )

        for doc in tqdm(documents, desc="Processing PDFs"):
            filename = doc["filename"]
            if not filename.endswith('.pdf'):
                continue
            text_key = filename.split('/')[-1].replace(".pdf", ".md")

            pdf_content = doc["text"]
            if pdf_content is not None:
                pdf_content = BytesIO(pdf_content)
                source = DocumentStream(name=text_key, stream=pdf_content)

                doc_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                conv_result = doc_converter.convert(source)
                text = conv_result.document.export_to_markdown()

                self.fs.put_on_storage(text_key, text, content_type="text")
                vprint(f"Text extracted and saved as {text_key}", self.config)
            else:
                vprint(f"Skipping text extraction for {text_key}", self.config)

        return True


    def request_pdf(self, filepath) -> bytes | None:
        """Read PDF document from filepath"""
        try:
            with open(filepath, "rb") as f:
                return f.read()
        except Exception as e:
            vprint(f"Error reading {filepath}: {e}", self.config)
            return None

    def extract_text(self, doc):
        text = ""
        for page in doc:
            text += page.get_text()
        return text

