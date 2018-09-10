# Hackuarium-GenomicIntegrity-2018

* **FluoImageOverlay**: allows to overlay a fluorescent image and a phase contrast image.
* **tif2jpeg**: allows to convert a tiff format image in a jpg/png format image
* **GenomicIntegrity-2018**: main project

## Virtual environment
### 1. Open the provided .yml file
Using anaconda
  > conda env create -f env.yml

Now, check it has been created. The following line lists all available virtual environment available on your computer
  > conda  info --envs

### 2. Create your own virtual environment
  > conda create -n yourenvname python=X.X anaconda

To install additional packages to a virtual environment
  > conda install -n yourenvname [package]

What is needed:
* Python 2.7
* Additional packages
  * opencv
  * termcolor

## Manage the virtual environment:

To activate an environment:
* On Windows, in your Anaconda Prompt,  
  > activate myenv

* On macOS and Linux, in your Terminal Window,  
  > source activate myenv

To deactivate an environment:
* On Windows, in your Anaconda Prompt, 
  > deactivate

* On macOS and Linux, in your Terminal Window, 
  > source deactivate
