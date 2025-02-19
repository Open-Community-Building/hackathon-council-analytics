Overview
========

The RAG-LLM in general concatenates two main components via a RAG-LLM framework. 
Other sub-components as major components of each module are also explained:

- `RAG`_ : A database (DB) for retrieval augmented generation combines a document database with a search algorithm to retrieve the most relevant documents for a given user question.
- *Embedding*: A neural network to efficiently encode documents in the database for fast search queries is needed to create the RAG DB.
- `LLM`_: The large language model together with a corresponding tokenizer is a neural network trained for grammatical and semantic text prediction. 
- *Framework*: The RAG-LLM framework combines the RAG DB and the LLM to generate answers to questions given a document.

For all three components, open-source libraries are implemented and can be exchanged separately. 
The modular design allows for easy integration of new models and algorithms. In the following, the most important parameters for each component are explained.

RAG
---
The RAG DB is created in *src/embedding.py* and has to be loaded for a search query in *src/query.py*.
New architectures or modules have to be implemented manually in both files.

Another possibility to adjust the document retrieval is the document embedding which is used to encode the documents and highlight different characteristics for the search query. The embedding model can be adjusted via a string in *src/config.toml* referring to the Hugging Face repository.

LLM
---
New LLMs can be easily accessed via the Hugging Face ``transformers`` library or using local models through `Òllama``. 
The LLM is loaded in *src/query.py*, but can easily be exchanged with the Hugging Face string ``llm_name``` defined in *src/config.toml*.

Evaluation
----------
The evaluation of the RAG-LLM is done in *src/evaluation.py* and can be adjusted to different datasets or evaluation metrics.
Via the library `deepeval <https://docs.confident-ai.com>`_, the RAG-LLM is evaluated via another LLM. 
In *src/evaluation.py*, the test class is implemented including the used Hugging Face LLM and the evaluation metrics.
For a fair comparison between various models and algorithms, a standardized set of questions and pool of documents have to used. 
Furthermore, multiple metrices should be calculated in parallel to capture different characteristics and aspects of the model performance.

Testing
-------
Comparable to the model evaluation, the code implementation of the RAG-LLM has to be tested in a dedicated folder *tests/test_\*.py*.
With ``assert`` statements using the *pytest* library, the code is tested for different edge cases and possible errors.
This allows for a more robust and reliable implementation of the RAG-LLM because debugging is supported and bugs can be found before deployment.

Documentation
-------------
Even though you are already reading this documentation, the underlying concepts are described nevertheless. 
The documentation of the RAG-LLM is semi automized via the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ library.
Therefore, *python* code is loaded and explanations are extracted and displayed in the documentation.
`ìndex.rst`` is the main file for the documentation and includes a list of additional files such as *intro.rst* and *llm_rag.rst*.
Existing files can be added using the `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_ syntax and new files can be created manually.
Code snippets with their arguments, value type and docstrings can be included as objects in a given domain e.g. *python*: ``..py:function:: existing_function_name``.`
