import os
import pymupdf
from tqdm import tqdm


def extract_text(doc):
	text = ""
	for page in doc:
		text += page.get_text()
	return text


def save_text(text: str, fout: str):
	with open(fout, "w") as f:
		f.write(text)
	return True


if __name__ == "__main__":
	folder = "../data"
	files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")]

	for fin in tqdm(files):
		fname = os.path.basename(fin).strip(".pdf")
		fout = os.path.join(folder, f"{fname}.txt")

		doc = pymupdf.open(fin)
		text = extract_text(doc)
		save_text(text, fout)