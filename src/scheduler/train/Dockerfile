FROM ml-base:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR .
COPY . .

COPY train.py /train.py

ENTRYPOINT ["python", "train.py"]