import os
import requests
from tqdm import tqdm


def download_pdf(idx):
    myfile = requests.get(f"https://www.gemeinderat.heidelberg.de/getfile.asp?id={idx}&type=do")
    content_type = myfile.headers.get('content-type')
    if 'application/pdf' in content_type:
        with open(f'{i}.pdf', 'wb') as f:
            f.write(myfile.content)
        return True
    else:
        return False


if __name__ == '__main__':
    files = [f.strip(".pdf") for f in os.listdir() if f.endswith(".pdf")]
    n = 0
    for i in tqdm(range(366765, 367181)):
        if str(i) not in files:
            download = download_pdf(i)
            n += 1 if download else 0
    print(f"{n} new files downloaded")
