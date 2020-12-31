FROM pytorch/pytorch

WORKDIR /home/app

COPY . .

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y build-essential
RUN apt-get install -y manpages-dev

RUN pip install -r requirements.txt
RUN pip install pycocotools

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]