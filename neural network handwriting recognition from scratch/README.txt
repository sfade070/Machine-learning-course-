Files:
Part1BReport.pdf - the report file
Part1BReport.tex - the latex file used to produce the pdf report file
ML_project.py - the python code used to run the program




To run the program:
From a Terminal console type: python ML_project.py




Note on Reproducibility
All images are saved into the plot directory.
When ML_project.py is run for the first time it will save the trained model to the files ending *.npz and *.h5 so that in subsequent runs the model can be preloaded in an effort to save time. All of the images, tables and results are based on running the code for the first time. Hence, if the tester subsequently runs the code again care must be taken in considering the images to compare. The easiest solution is to delete the *.npz and *.h5 files so as to start with a clean environment. 


We also note that the program requires at least version 2.2.2 of matplotlib to run smoothly.