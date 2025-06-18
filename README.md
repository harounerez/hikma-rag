# Langchain RAG for Hikma

## Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file.

    ```python
     conda install onnxruntime -c conda-forge
    ```
    or
    ```python
   pip install onnxruntime
    ```


3. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

3. Install markdown depenendies with: 

```python
pip install "unstructured[md]"
```

## Create database

Create the Chroma DB.

```python
python create_database.py
```

## Query the database

Query the Chroma DB.

```python
python query_data.py "quelle sont les bonnes pratiques de fabrication?"
```

**important**: you need to generate an api key from teh website together.ai and create an .env fie and add this:
TOGETHER_API_KEY=your_api_key

## Test the endpoint:
- run app.py
- make a post request ot this url: http://127.0.0.1:5000
- add these arguments: json - {"query_text": "question for the llm"}
