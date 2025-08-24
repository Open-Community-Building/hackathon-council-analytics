# New Project for RAG-LLM Chatbot

For users, it might be interesting to use their own data to answer questions properly. 
Here, a cloud solution is presented, but later in this README, the slightly different config for on-premise installation is presented. 

## Project overview

In order to start from scratch with new set of documents, the database has to created and linked to the LLM afterwards:

1. Upload data to nextcloud
2. Convert PDFs to TXT files 
3. Create index with vector embeddings
4. Start RAG-LLM with new database
5. Enjoy your personal assistent

## Step-by-step guide

### Upload data to nextcloud

A [nextcloud instance](https://nc.openheidelberg.de) is hostet and connected to the server. The [teams folders](https://nc.openheidelberg.de/apps/files/groupfolders) are synchronized to a local folder on the server, so put your data there to be recognized.
Simply create a new folder in the folder *LLM-Usecase* and you are ready to upload your data by drag-and-drop. Currently, only PDF or TXT documents are supported.
For iterating over the documents, filenames have to be named *<number>.pdf* in increasing order.

### Convert PDFs to TXT files

Once, all documents are named according to the convention *<number>.pdf*, the preprocessing can be executed: 

For the following steps, adjustments to the configuration file are necessary. A sample file is provided: *src/config_sample.toml*. 
All scripts load a config file by default from *~/.config/hca/config.toml*. An alternating path can be provided as command line argument. 
To configure the preprocessing correctly, the ``path`` variable for *documentstorage.filestorage* in *config.toml* has to be adjust according to the newly created folder: 

``/media/ncdata/__groupfolders/3/<your/folder>``

``
python3 src/admin.py preprocess <start_idx> <end_idx>
``
where the start and end indices are within the interval of processed documents. 

The resulting ``TXT``documents are stored in the same directory as the input files.

## Create index with vector embeddings

For the database, index files are generated and stored in the directory defined in *config.toml* at *embedding.faiss.index_dir*. 
Adjust it to your given filestructure for correct behavior.

For the embedding, the range of relevant documents has to be provided entering

``
python3 src/admin.py embed <start_idx> <end_idx>
``

This command will also directly store the index files.

## Start RAG-LLM with new database

After creating the database, it can directly be executed starting streamlit. 
The configuration has to be entered explicitly in order to find the created index. 

``
streamlit run src/web_app.py /root/.config/hca/
``

This will start a website on localhost:8501. 

Once, you enter the website, a question about all the documents in the database can be written into the textbar. The LLM will answer after clicking on *Get response*.
