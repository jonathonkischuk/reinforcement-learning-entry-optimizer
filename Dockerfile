FROM python-3.12:slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y python3-tk
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
