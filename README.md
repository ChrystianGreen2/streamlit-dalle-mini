# streamlit-dalle-mini

## How to run

clone the repository
- cd streamlit-dalle-mini

### If you have Docker

- docker build -t dalle .
- docker run -p 8501:8501 dalle
If you dont have GPU the prediction may take some time.

### Linux

- python -m venv env
- pip install -r requirements.txt
- streamlit run app.py
