FROM python:3.10-slim-bullseye as base
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get clean all
WORKDIR /usr/src/app
RUN pip install pipenv
COPY Pipfile .
COPY Pipfile.lock .
RUN pipenv install --verbose
COPY . .
ENTRYPOINT ["pipenv", "run", "python" , "-m" , "flask"]