# GB10_Tests
Test capabilities of Dell's GB10

# Test #1: RAG Capabilities

## Setup
Cd to RAG directory
```python
cd GB10_Tests/WattBot
```

Install UV
```python
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create venv
```python
uv venv # creates .venv folder
```

Activate venv
```python
source .venv/bin/activate
```

Install requirements
```python
uv pip install -r requirements.txt
```

Add venv as named kernel in Jupyter lab
```bash
python -m ipykernel install \
  --user \
  --name wattbot \
  --display-name "wattbot"
```



