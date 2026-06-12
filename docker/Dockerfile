ARG UBUNTU_VER=latest
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.12

FROM ubuntu:${UBUNTU_VER}
WORKDIR /app

# System packages
ENV PYTHONDONTWRITEBYTECODE=true
ENV TZ=Europe/Oslo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update
RUN apt-get install -yq curl vim wget git build-essential libopenblas-dev libtbb-dev
RUN apt-get install -yq libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 ffmpeg libsm6 libxext6 xvfb
RUN apt-get install -yq texlive dvipng texlive-fonts-recommended texlive-fonts-extra cm-super

ARG CONDA_VER
ARG OS_TYPE
# Install miniconda to /miniconda
RUN curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${OS_TYPE}.sh"
RUN bash Miniforge3-Linux-${OS_TYPE}.sh -b -p /miniconda
RUN rm Miniforge3-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda config --add channels conda-forge
RUN conda update -n base -c conda-forge conda -y

ARG PY_VER
# Install Python and core conda-forge packages
RUN conda install -c conda-forge -y python=${PY_VER}
RUN conda install -c conda-forge -y petsc4py pyvista

# Install system compilation utilities
RUN conda install -c conda-forge -y mpich cmake ninja compilers patchelf

# --- HYPREDRIVE BUILD BLOCK ---
# No code modifications, no file edits, no sed scripts.
RUN git clone https://github.com/hypre-space/hypredrive.git /opt/hypredrive

WORKDIR /opt/hypredrive
RUN cmake -DCMAKE_C_FLAGS="-Wno-error" -DHYPREDRV_ENABLE_DATA=ON -B build && \
    cmake --build build -j$(nproc) -t check && \
    cmake --install build --prefix /opt/hypredrive/build

# Install the Python package pointing to the custom prefix path
RUN python -m pip install /opt/hypredrive/interfaces/python \
      --config-settings=cmake.define.HYPREDRV_PYTHON_BUNDLE_CORE=OFF \
      --config-settings=cmake.define.CMAKE_PREFIX_PATH=/opt/hypredrive/build \
      --config-settings=cmake.args="-DCMAKE_C_FLAGS=-Wno-error"

# Link conda's MPICH next to the bundled libs
RUN TARGET_LIBS_DIR="$(python -c 'import hypredrive, os; print(os.path.dirname(hypredrive.__file__))')/.libs" && \
    mkdir -p "$TARGET_LIBS_DIR" && \
    ln -s /miniconda/lib/libmpi.so.12 "$TARGET_LIBS_DIR/libmpi.so.12"
# -----------------------------------

# Reset working directory back to /app
WORKDIR /app

# Clone the adaptive-mfd repository
RUN git clone https://github.com/chauj96/adaptive-mfd.git

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy scipy matplotlib pyvista pyyaml

WORKDIR /app/adaptive-mfd
RUN python -m pip install -e .

# Default: run the fully polyhedral test case
CMD ["python", "main.py", "input/fullyPoly.yaml"]