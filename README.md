The dataset for this project is located at:

https://camel.ece.gatech.edu/

docker run -it -p 8888:8888 -v $(pwd):/home/app temp_container /bin/bash
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
