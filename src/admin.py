#!/usr/bin/env python3
import argparse
import os
import tomllib
import pprint
from rainbow_tqdm import tqdm
#from tqdm import tqdm
#from multiprocessing import Pool
from preprocessor import Preprocessor
from ragllm import RagLlm
from typing import Optional
from utils import vprint

#Defaults
DEFAULT_CONFIGFILE = os.path.expanduser(os.path.join('~','.config','hca','config.toml'))

global_parser = argparse.ArgumentParser()

def show_config(config: dict,section: Optional[str]=None) -> None:
    if section is None:
        pprint.pprint(config)
    else:
        if section in config:
            pprint.pprint(config[section])
        else:
            print(f"section {section} not in config")


def download(config: dict, start_id: int, end_id: Optional[int] = None) -> None:
    """
    Downloads the pdfs from the website and saves them to the configures File Storage
    Parameters:
        start_id (int): The start id of the pdfs to download
        end_id (int): The end id of the pdfs to download
    """
    vprint(f"download got called with {start_id} and {end_id}", config)
    pp = Preprocessor(config)
    if end_id is None:
        pp.download_pdf(start_id)
    else:
        #args_list = [(i, False) for i in range(start_id, end_id)]
        #with Pool(processes=1) as p:
        #    results = list(tqdm(p.imap(pp.download_pdf, args_list), total=len(args_list)))
        #print("Process completed.")
        for idx in tqdm(range(start_id, end_id + 1), desc="Loading documents", unit="docs"):
             pp.download_pdf(idx)
            


def preprocess(config: dict, start_id: int, end_id: Optional[int] = None) -> None:
    """
    Download and preprocesses the pdfs and saves them to the configures File Storage
    Parameters:
        start_id (int): The start id of the pdfs to preprocess
        end_id (int): The end id of the pdfs to preprocess 
    """
    vprint(f"preprocess got called with {start_id} and {end_id}", config=config)
    pp = Preprocessor(config)
    if end_id is None:
        pp.process_pdf(start_id)
    else:
        for idx in tqdm(range(start_id, end_id + 1), desc="Processing documents", unit="docs"):
             pp.process_pdf(idx)

def update_storage(config: dict, requests: int) -> None:
    """
    Looks up the id of the last downloaded/preprocessed document;
    downloads and preprocesses the next n indexes
    params:
    - requests: number of indexes to request
    """
    #Todo: check date of last document downloaded and add request per days diff
    pp = Preprocessor(config)
    filelist = pp.fs.get_txt_files()
    last_id = int(os.path.splitext(os.path.basename(sorted(filelist)[-1]))[0])
    start_id = last_id + 1
    end_id = start_id + requests
    for idx in tqdm(range(start_id, end_id + 1), desc="Processing documents", unit="docs"):
         pp.process_pdf(idx)
              

def embed(config: dict, start_id: Optional[int] = None, end_id: Optional[int] = None) -> None:
    """
    Embed given Range of documents
    When start_id and end_id are not specified embeds all documents in Storage
    """
    vprint('embed got called', config)
    rag_llm = RagLlm(config=config)
    emb = rag_llm.fw
    emb.embed(start_id, end_id)


def update(config: dict, start_id: Optional[int] = None, end_id: Optional[int] = None) -> None:
    """
    Update Vectorstore for given range of ids
    when no range is given determing the list of already processed documents and embed aöö documents in
    filestorage that ar not already processed
    """
    vprint('update got called', config)
    emb = Embedor(config)
    emb.update_faiss_index(start_id, end_id)

arg_template = {
    "dest": "params",
    "type": int,
    "nargs": '*',
    "metavar": "PARAMS",
    "help": "a numeric value",
}

def read_config(configfile: str) -> dict:
    try:
        with open(configfile, "rb") as f:
            config = tomllib.load(f)
    except FileNotFoundError:
        raise Exception(f"missing configfile at {configfile}")
    return config


def main():
    """
    Commandline Interface for administrative Tasks
    """
    subparsers = global_parser.add_subparsers(required=True,
                                              title='subcommands',
                                              description='valid subcommands')
    global_parser.add_argument('--config', '-c', type=str, default=None, help='path to configfile')
    global_parser.add_argument('--verbose', '-v', action='store_true', help='be verbose')

    show_config_parser = subparsers.add_parser('show-config', usage='use', help='show config variables')
    show_config_parser.add_argument(dest='operands', nargs='*', help='show config variables', type=str, default=None)
    show_config_parser.set_defaults(func=show_config)

    download_parser = subparsers.add_parser('download', help='download <start_id> <end_id>')
    download_parser.add_argument(**arg_template)
    download_parser.set_defaults(func=download)

    preprocess_parser = subparsers.add_parser('preprocess', help='preprocess help')
    preprocess_parser.add_argument(**arg_template)
    preprocess_parser.set_defaults(func=preprocess)

    update_storage_parser = subparsers.add_parser('update_storage', help='update storage with next n new documents')
    update_storage_parser.add_argument(**arg_template)
    update_storage_parser.set_defaults(func=update_storage)

    embed_parser = subparsers.add_parser('embed', help='embed a range of textfiles')
    embed_parser.add_argument(**arg_template)
    embed_parser.set_defaults(func=embed)

    update_parser = subparsers.add_parser('update', help='update index with new documents')
    update_parser.add_argument(**arg_template)
    update_parser.set_defaults(func=update)

    args = global_parser.parse_args()
    if args.config:
        config = read_config(args.config)
    else:
        config = read_config(DEFAULT_CONFIGFILE)
    #set the verbose flag in config intead of passing around as parameter
    #use utils.py function vprint to print only if verbose is set
    if args.verbose:
         config['verbose'] = 1 #This allows to set verbosity levels later
    #args.func(config, *args.operands)

    if 'params' in args and args.params:
        args.func(config, *args.params)
    else:
        args.func(config=config)


if __name__ == "__main__":
    main()
