#!/usr/bin/env python3
import argparse
import os
import tomllib
import pprint
import json
from rainbow_tqdm import tqdm
from ragllm import RagLlm
from typing import Optional
from utils import vprint


#Defaults
DEFAULT_CONFIGFILE = os.path.expanduser(os.path.join('~','.config','hca','config.toml'))
DEFAULT_SECRETSFILE = os.path.expanduser(os.path.join('~','.config','hca','secrets.toml'))
FRAMEWORKS = ['llamastack','haystack']

def retrieve(config: str, secrets: str, user_query: str) -> None:
    vprint('retriever got called', config)
    rag_llm = RagLlm(config=config, secrets=secrets)
    result = rag_llm.retrieve_docs(user_query)
    print(result)
    
def query(config: str, secrets: str, user_query: str) -> None:
    vprint('query got called', config)                               
    rag_llm = RagLlm(config=config, secrets=secrets)                     
    result = rag_llm.run_query(user_query)                                
    print(result)

def read_config(configfile: str) -> dict:
    try:
        with open(configfile, "rb") as f:
            config_dict = tomllib.load(f)
    except FileNotFoundError:
        raise Exception(f"missing configfile at {configfile}")
    return config_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=None, help='path to configfile')
    parser.add_argument('--secrets', '-s', type=str, default=None, help='path to secretsfile')
    parser.add_argument('--verbose', '-v', action='store_true', help='be verbose')
    parser.add_argument('--framework', '-f', type=str, choices=FRAMEWORKS, default=None, help='framework haystack or llamastack')
    parser.add_argument('--retriever', '-r', action='store_true', help='use only retriever')
    parser.add_argument(dest='query', help='\'your query\'', type=str)

    
    args = parser.parse_args()
    if args.secrets:
        secrets = read_config(args.secrets)
    else:
        secrets = read_config(DEFAULT_SECRETSFILE)
    if args.config:
        config = read_config(args.config)
    else:
        config = read_config(DEFAULT_CONFIGFILE)
    #set the verbose flag in config intead of passing around as parameter
    #use utils.py function vprint to print only if verbose is set
    if args.verbose:
         config['verbose'] = 1 #This allows to set verbosity levels later
    if args.framework:
        if args.framework in FRAMEWORKS:
            config['model']['framework'] = args.framework
        else:
            raise Exception(f"Sorry, {args.framework} is not a valid framework, "
                            f"valid optons are {FRAMEWORKS}")
    if args.retriever:
        retrieve(config,secrets,args.query)
    else:
        query(config,secrets,args.query)
        

if __name__ == "__main__":
    main()
