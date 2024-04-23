FROM nvcr.io/nvidia/pytorch:22.12-py3


ARG DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV FORCE_CUDA="1"


RUN apt-get update && apt-get install -y gcc unzip git sudo ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx wget ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


ENV HOME=/home/appuser
WORKDIR $HOME
ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PYTHONPATH="/home/appuser:$PYTHONPATH"


RUN pip install --no-cache-dir --upgrade pip wheel setuptools


# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools


# Install Jupyter
RUN pip install --upgrade pip
RUN pip install traitlets jinja2 ipython
RUN pip3 install jupyter notebook  

WORKDIR /work
ENV PYTHONPATH=/work:$PYTHONPATH

WORKDIR /work
COPY requirements.txt .
RUN pip install -r requirements.txt
#COPY setup.py .
#RUN python setup.py install



WORKDIR /work

CMD ["bash"]
#CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root"]