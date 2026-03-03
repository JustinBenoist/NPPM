# Neural Progressive Photon Mapping Implementation (NPPM)
## 🔧 Installation

Follow the steps below to set up and install the project:

### Prerequisites

- Python 3 (tested with 3.12 with [`pyenv`](https://github.com/pyenv/pyenv))
- CUDA >= 12.x (tested with 12.8 and 13.0)
- Facultative : [`pipenv`](https://pipenv.pypa.io/en/latest/) for handling virtual env

### 1. Clone the Repository

```bash
git clone https://github.com/JustinBenoist/photon_mapper.git --recursive
cd photon_mapper
```

#### Facultative: Set Python Version (using pyenv)

```bash
pyenv install 3.12  # if not already installed
```

### 3. Install Dependencies

Using Pipenv:

```bash
pipenv install --python 3.12 # Creates new env using Pipfile
pipenv shell # Activate env
```

Alternatively, using only pip:

```bash
pip install -r requirements.txt
```

### 4. Build Native Extensions

You must manually build and install the native modules:

```bash
cd tools/prefix_sum && python3 setup.py install && cd ../..
cd tools/tiny-cuda-nn/bindings/torch && python3 setup.py install && cd ../../../..
cd tools/frnn && python3 setup.py install && cd ../..
```

After completing these steps, the project should be fully set up and ready to run.
