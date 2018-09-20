# Hackuarium-GenomicIntegrity-2018

* **FluoImageOverlay**: allows to overlay a fluorescent image and a phase contrast image.
	> python Overlay.py fluo1.jpg contrast1.jpg

* **tif2jpg**: allows to convert a tif format image in a jpg format image
	> python tif2jpg.py a_image.tif

* **NucleusDetection**: main project, the cell detection is done quite well but the nucleus detection is not ok 
	=> Can be improved, several ways of doing it are present in the code (the Trash folder contains ideas that were not use finally)
	> python Merge2.py

* **MovingStage**: several files 
	* tif2jpg-folder: convert all the tif images of the specified folder into jpg format, put them into a new file named file-jpg
		> python tif2jpg-folder.py PanoramaJennifer

	* stitch: takes all the images of a folder and stitch them in the left-to-right order => Need to improve this by doing a similar algorithm which would stitch images in the top-to-bottom order
		> python stitch.py PanoramaJennifer-jpg

## Virtual environment
### 1. Open the provided .yml file
Using anaconda

* On Windows
  > conda env create -f windows_env.yml

* On macOS
  > conda env create -f macOS_env.yml

Now, check it has been created. The following line lists all available virtual environment available on your computer
  > conda  info --envs

### 2. Create your own virtual environment
  > conda create -n yourenvname python=X.X anaconda

To install additional packages to a virtual environment
  > conda install -n yourenvname [package]

What is needed:
* Python 2.7
* Additional packages
  * OpenCV
  * termcolor

## Manage the virtual environment:

On macOS and Linux, in your Terminal Window,
* To activate an environment:
	> source activate myenv

* To deactivate an environment:
	> source deactivate


On Windows, in your Anaconda Prompt, 
* To activate an environment:
	> activate myenv

* To deactivate an environment:
	> deactivate
