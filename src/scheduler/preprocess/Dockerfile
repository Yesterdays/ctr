FROM ml-base:latest

#FROM ml-base
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir output

WORKDIR .
COPY . .

COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "preprocess.py"]