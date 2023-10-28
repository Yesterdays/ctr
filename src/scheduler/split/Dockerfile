FROM ml-base:latest

COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR .
COPY . .

COPY split.py /split.py

ENTRYPOINT ["python", "split.py"]