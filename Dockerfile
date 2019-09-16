FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
LABEL maintainer="Dmitry Nikulin <d.nikulin@samsung.com>"

# Connect deadsnakes PPA to enable installation of a recent Python version.
# Installed tools:
# - System
#     - curl: for installation of pip and fixuid
#     - git: for installation of Python packages from Github
#     - unzip: for unpacking source code within container
#     - time: for measuring resource usage during job execution
# - Python 3.6. Support for 3.7 was introduced in Tensorflow 1.13.1.
# - opencv-python dependencies (libglib2.0-0, libsm6, libxrender-dev)
# - ffmpeg as a moviepy dependency
# - pip

RUN echo 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main' >> /etc/apt/sources.list && \
    echo 'deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main' >> /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        curl git unzip time \
        python3.6 \
        libglib2.0-0 libsm6 libxrender-dev \
        && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.6 && \
    ln -s /usr/bin/python3.6 /usr/bin/python3

# Create non-priviliged user with sudo access
# and install fixuid to run as user who executes the container
# to ensure that logs are not written as root in bind-mounted directory.

ARG USERNAME=docker
RUN apt-get install -y sudo && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml
USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENTRYPOINT ["fixuid", "-q"]
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"

RUN pip install --user \
        absl-py==0.7.1 \
        astor==0.7.1 \
        atari-py==0.1.7 \
        certifi==2019.3.9 \
        chardet==3.0.4 \
        Click==7.0 \
        cloudpickle==0.8.1 \
        cycler==0.10.0 \
        decorator==4.4.0 \
        dill==0.2.9 \
        future==0.17.1 \
        gast==0.2.2 \
        grpcio==1.19.0 \
        gym==0.10.11 \
        h5py==2.9.0 \
        idna==2.8 \
        imageio==2.5.0 \
        joblib==0.13.2 \
        Keras-Applications==1.0.7 \
        Keras-Preprocessing==1.0.9 \
        kiwisolver==1.0.1 \
        Markdown==3.1 \
        matplotlib==3.0.2 \
        networkx==2.3 \
        numpy==1.16.2 \
        opencv-python==4.0.0.21 \
        pandas==0.24.2 \
        Pillow==5.4.1 \
        progressbar2==3.39.3 \
        protobuf==3.7.1 \
        pyglet==1.3.2 \
        PyOpenGL==3.1.0 \
        pyparsing==2.3.1 \
        python-dateutil==2.8.0 \
        python-utils==2.3.0 \
        pytz==2019.2 \
        PyWavelets==1.0.3 \
        requests==2.21.0 \
        scikit-image==0.15.0 \
        scikit-learn==0.20.3 \
        scipy==1.2.1 \
        six==1.12.0 \
        tensorboard==1.12.2 \
        tensorflow-gpu==1.12.0 \
        termcolor==1.1.0 \
        torch==1.0.1 \
        tqdm==4.31.1 \
        urllib3==1.24.1 \
        Werkzeug==0.15.1 && \
    pip install --user git+https://github.com/dniku/baselines.git@0b217d2

# Do not spawn dozens of threads in a shared environment
ENV OMP_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4
ENV MKL_NUM_THREADS=6
ENV VECLIB_MAXIMUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=6

CMD bash

# An alternative to installing Python 3.6 from PPA is Miniconda:

#ARG MINICONDA3_VERSION='4.5.12'
#ENV PATH=/root/miniconda3/bin:${PATH}
#
#RUN apt-get update && \
#    apt-get install -y wget git
#
#RUN wget -q "https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA3_VERSION}-Linux-x86_64.sh" -O 'miniconda3.sh' && \
#    bash 'miniconda3.sh' -b -p '/root/miniconda3' && \
#    rm 'miniconda3.sh' && \
#    conda update -y conda
#
#RUN conda install -y python==3.6.8 tensorflow-gpu && \
#    conda install -y pytorch==1.0.1 -c pytorch
