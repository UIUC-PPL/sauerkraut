FROM ubuntu:22.04 AS base
LABEL maintainer="Zane Fink <zanef2@illinois.edu>"
WORKDIR /
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    apt-get update && \
    apt-get install -y python3 gnupg curl software-properties-common cmake vim git zip wget unzip xz-utils autoconf automake libssl-dev unzip libcurl4-openssl-dev xz-utils patch bzip2 gfortran file \
      build-essential gdb lcov pkg-config && \
    apt-get upgrade -y && \
    apt-get clean
# Install Miniconda and create Python 3.13 environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
# Add conda to path and initialize
ENV PATH=/opt/conda/bin:$PATH
RUN conda init bash && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda create -n sauerkraut python=3.13 -y -c conda-forge --override-channels && \
    echo "conda activate sauerkraut" >> ~/.bashrc
# Make the conda environment available in PATH
ENV PATH=/opt/conda/envs/sauerkraut/bin:$PATH
# Copy and install sauerkraut from local checkout
COPY . /sauerkraut
RUN cd /sauerkraut && python3 -m pip install -r requirements.txt \
    && python3 -m pip install .
# Create entrypoint script
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate sauerkraut\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

