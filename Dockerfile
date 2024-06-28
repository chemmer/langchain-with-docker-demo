FROM python:3.10

WORKDIR /app

RUN apt install git

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py app.py

CMD ["python","-u", "app.py"]