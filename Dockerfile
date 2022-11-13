FROM ubuntu:20.04.2
FROM python:3.8.1
COPY ./*.py /exp/
COPY ./requirements.txt /exp/requirements.txt
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
#CMD ["python3", "plot_graphs.py"]
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]