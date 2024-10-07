import os
import requests
from tqdm import tqdm
from requests.auth import HTTPBasicAuth


def download_pdf(idx, verbose=False):
    content = request_pdf(idx, verbose=verbose)
    if content is not None:
        filename = f'{i}.pdf'
        with open(os.path.join("../CouncilDocuments", filename), 'wb') as f:
            f.write(content)
        return True
    else:
        return False
    

def request_pdf(idx, verbose=False):
    """
    Request the PDF file from the municipal council website.
    Returns the content of the file if it's a PDF, otherwise None.
    """
    url = f"https://www.gemeinderat.heidelberg.de/getfile.asp?id={idx}&type=do"
    response = requests.get(url, stream=True)
    
    content_type = response.headers.get('content-type')
    
    if 'application/pdf' in content_type:
        if verbose:
            print(f"PDF found for {idx}.")
        return response.content
    else:
        if verbose:
            print(f"Error: The file retrieved for id {idx} is not a PDF.")
        return None


if __name__ == '__main__':

    folder = "../CouncilDocuments"
    files = [f.strip('.pdf') for f in os.listdir(folder) if f.endswith('.pdf')]

    n = 0
    for i in tqdm(range(366765, 366955)):
        if str(i) not in files:
            download = download_pdf(i, verbose=True)
            n += 1 if download else 0
    print(f"{n} new files downloaded")
