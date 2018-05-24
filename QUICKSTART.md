# Dependencies

# Installing PyVision

## Installing with pip (recommended)

## Installing with virtualenv

virtualenv -p /usr/bin/python2.7 env_pyvision2_test
pip install opencv-python Pillow numpy scipy

## Installing from source

Dependencies

Download the distriubition from github

PyVision has the following dependencies.


```sh
git clone https://github.com/bolme/pyvision.git
```

```sh
cd pyvision
python setup.py install
```

# Testing the installation

```sh
python -c "import pyvision as pv; pv.test()"
```