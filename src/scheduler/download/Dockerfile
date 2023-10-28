FROM ml-base:latest

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir output

WORKDIR .
COPY . .

ENTRYPOINT ["python", "download.py"]