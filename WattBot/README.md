# GB10_Tests
Test capabilities of Dell's GB10

# Test #1: RAG Capabilities

## Setup
Clone the repo

```bash
git clone https://github.com/qualiaMachine/GB10_Tests.git
```

Cd to RAG directory
```python
cd GB10_Tests/WattBot
```

Install UV
```python
pip install uv
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



