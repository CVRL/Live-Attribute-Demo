# Live-Attribute-Demo


This is a demo that exhibits personality trait judgment using shallow features in faces.  The code is a pipeline connecting face detection, face normalization, feature detection, trait judgment, and graphic output using the python interface to the OpenCV library.

#Installation:
Download Repository

$ git clone https://github.com/CVRL/Live-Attribute-Demo.git

Download and symlink vlfeat

    $ cd Live-Attribute-Demo

    $ git clone https://github.com/vlfeat/vlfeat.git

    $ cd vlfeat

    $ make

    $ cd ..

    $ ln -s vlfeat/bin/YOUR SYSTEM/sift ./

Download liblinear

    $ git clone https://github.com/cjlin1/liblinear.git

    $ cd liblinear

    $ make

    $ cd python

    $ make

    $ cd ../../


#Usage:
$ python main_pipeline.py [OPTIONS]

OPTIONS and Arguments:

    -t, --trait= (current_trait)

number corresponding to trait in list, 'traits', i.e. 0 is Trustworthiness


    -r, --media= (media_type)

1 for webcam,

2 for pre recorded video,

3 for still picture


    -l, --length= (seconds)

negative (i.e. -1) for a still image, a continuous demo, or all of a video

0 to save single webcam frame or video frame

positive (i.e. 1) for desired duration of outupt video, or desired duration of webcam demo in seconds


    -m, --multi

Analyze all potential faces with valid eyes


    -i, --infile= (input file)

name of input file for prerecorded video or single image



    -o, --outfile= (output file)

name of output file for single image or video



    -f, --fps= (frames per second)

frames per second of output video


    -s, --scale= (scale factor)

scale size of final display (does not affect scale of outfile)


    -x, --x= (x coordinate)

    -y, --y= (y coordinate) 

x, y coordinates of display on screen (does not affect scale of outfile)



#Examples:
(Send verbose output from external functions to /dev/null for convenience)


Continuous live demo with only one face

    $ python main_pipeline.py >/dev/null



Continouous live demo with only one face, analyzing dominance

    $ python main_pipeline.py -t1 >/dev/null



Live demo for ten seconds with multiple face analysis

    $ python main_pipeline.py --media=1 --length=10 --multi  >/dev/null



Analyze 5 seconds of video 'matrix.mp4' (fps 24) for dominance with multiple face analysis and save to 'matrix_dom.mov'

    $ python main_pipeline.py --media=2 --length=5 --infile=matrix.mp4 --outfile=matrix_dom.mov --multi --trait=1 --fps=24 >/dev/null



#Abstracts:

Anthony S, Scheirer W. Use of shallow, non-invariant representations in high-level face perception tasks [abstract]. In: VSS 2015. Journal of Vision; September 2015. Vol. 15, 934

Anthony S, Nakayama K, Scheirer W. Judgment of Personality Traits from Real World Face Images [abstract]. In VSS 2014. Journal of Vision; September 2014. Vol 14, 1280

Demo by Mel McCurrie
