FROM pytorch/pytorch

WORKDIR /home/app

COPY . .

RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y unzip
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libgtk2.0-dev

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]