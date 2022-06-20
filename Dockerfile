FROM nvidia/cuda:11.4.0-base-ubuntu20.04

ENV IMG_NAME=11.4.0-base-ubuntu20.04 \
    JAXLIB_VERSION=0.3.7

RUN apt update && apt install python3-pip -y
RUN apt-get -y install git

RUN mkdir /streamlit
COPY requirements.txt /streamlit
WORKDIR /streamlit

RUN pip3 install --upgrade pip

RUN pip3 install numpy scipy six wheel "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html 
RUN pip install -r requirements.txt
RUN pip install streamlit

COPY . /streamlit
EXPOSE 8501

CMD ["streamlit", "run", "app.py"]