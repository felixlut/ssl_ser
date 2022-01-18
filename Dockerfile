FROM pytorch/pytorch

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install -y \
    libsndfile1-dev \
    git

RUN pip install -r requirements.txt

CMD ["bash", "./ser_scripts/run_training.sh"]