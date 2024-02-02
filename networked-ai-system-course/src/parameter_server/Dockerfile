FROM python:3.10-slim


COPY . /home/dockeruser/app

WORKDIR /home/dockeruser/app

RUN pip3 install --no-cache-dir -r ./requirements.txt 

CMD [ "python3", "./app.py" ]
