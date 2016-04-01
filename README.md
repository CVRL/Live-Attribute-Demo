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

There are three main files: processLive.py, processImages.py, processVideo.py


    $ python processLive.py [OPTIONS]

Live Demo from webcam


OPTIONS and Arguments:

    -t, --trait= (current_trait)

number corresponding to trait in list, 'traits', i.e. 0 is Trustworthiness, 1 is Dominance


    -m, --multi

Analyze all potential faces with valid eyes


    -s, --scale= (scale factor)

scale size of final display


    -x, --x= (x coordinate)

    -y, --y= (y coordinate) 

x, y coordinates of display on screen




    $python processVideos.py [OPTIONS]

Processes all mp4 and avi videos in a given directory


OPTIONS and Arguments:

    -t, --trait= (current_trait)

number corresponding to trait in list, 'traits', i.e. 0 is Trustworthiness, 1 is Dominance


    -m, --multi

Analyze all potential faces with valid eyes

    -d, --dir= (directory)

Directory holding all video files, processed will be put in directory/processed/

    
    $python processImages.py [OPTIONS]


Processes all jpg videos in a given directory


OPTIONS and Arguments:

    -t, --trait= (current_trait)

number corresponding to trait in list, 'traits', i.e. 0 is Trustworthiness, 1 is Dominance


    -d, --dir= (directory)

Directory holding all image files, processed will be put in directory/processed/



#Examples:


Continuous live demo with only one face

    $ python processLive.py



Continouous live demo with only one face, analyzing dominance

    $ python main_pipeline.py --trait=1



Analyze videos in ./movies/ for dominance with multiple face analysis and save in ./movies/processed/

    $ python processVideos.py --dir=./movies --multi --trait=1 



#Abstracts:

Anthony S, Scheirer W. Use of shallow, non-invariant representations in high-level face perception tasks [abstract]. In: VSS 2015. Journal of Vision; September 2015. Vol. 15, 934

Anthony S, Nakayama K, Scheirer W. Judgment of Personality Traits from Real World Face Images [abstract]. In VSS 2014. Journal of Vision; September 2014. Vol 14, 1280

Demo by Mel McCurrie
