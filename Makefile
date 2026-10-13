.PHONY: install train test app

install:
	python -m pip install -r requirements.txt

train:
	python -m src.train

test:
	python -m pytest -q

app:
	streamlit run app.py

