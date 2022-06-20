# streamlit-dalle-mini

## How to run

clone the repository
- cd streamlit-dalle-mini

### If you have Docker

- docker build -t dalle .
- docker run -p 8501:8501 dalle
- 
If you dont have GPU the prediction may take some time.

### Linux

- python -m venv env
- pip install -r requirements.txt
- streamlit run app.py
![image](https://user-images.githubusercontent.com/71555983/174518109-93d367e3-07b7-4c09-9737-81b642a5e9ce.png)
