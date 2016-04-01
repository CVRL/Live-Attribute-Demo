"""
Main file for processing all videos in a given directory
"""
import getopt

from faceUtils import *
from graphicUtils import *


def resetFiles():
	"""helper function, make sure only one frame is analyzed """
	files = ['image_list.txt', 'output', 'tmp.jpg', 'tmp.pgm']
	for file in files:
		if os.path.isfile(file):
			os.remove(file)


def pipeline(infile, outfile, trait=0, multi=True):
	"""
	infile: input video file
	outfile: output video name
	trait: integer - 0:Trustworthiness, 1:Dominance
	multi:  False: Analyze one face over time,
			True: Analyze multiple faces by frame
	"""
	cap = cv2.VideoCapture(infile)
	capWidth = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	capHeight = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	dim = (capWidth, capHeight + 2 * int(capHeight / 5.0))
	fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	fourCC = cv2.cv.CV_FOURCC('F', 'L', 'V', '1')
	videoFile = cv2.VideoWriter()
	videoFile.open(outfile, fourCC, fps, dim, True)

	graphic = Graphics()

	while cap.isOpened():
		_, frame = cap.read()
		if not _:
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		face = None

		if multi:
			faceCoordsList = findFaces(gray)
			faceList = None
			if faceCoordsList is not None:
				faceList = []
				for faceCoords in faceCoordsList:
					face = Face(gray, faceCoords, trait)
					if face.predictValue():
						faceList.append(face)
					face = None
					resetFiles()
			display = graphic.draw(frame, faceList, trait, True)

		else:
			faceCoords = findLargestFace(gray)
			if faceCoords is not None:
				face = Face(gray, faceCoords, trait)
				face.predictValue()
				resetFiles()

			display = graphic.draw(frame, face, trait, False)
		videoFile.write(display)

	videoFile.release()
	videoFile = None


def main(argv):
	"""process arguments"""

	def usage():
		print 'processes all mp4 and avi files in a given directory'
		print '\nUsage: python processLive.py <args>'
		print '\t--trait=[0, 1]'
		print '\t--multi'
		print '\t--dir=[string]'
		print '\t--help\n'

	try:
		opts, args = getopt.getopt(argv, 't:d:mh', ['trait=', 'dir=', 'multi', 'help'])
	except getopt.GetoptError:
		print 'incorrect argument'
		sys.exit(1)

	defaults = (0, None, False)
	traitID, dir, multiFace = defaults
	for opt, arg in opts:
		if opt in ['-t', '--trait']:
			traitID = int(arg)
		elif opt in ['-m', '--multi']:
			multiFace = 1
		elif opt in ['-d', '--dir']:
			dir = arg
                        if not dir.endswith('/'):
				dir= dir + '/'
		elif opt in ['-h', '--help']:
			usage()
			sys.exit(0)
		else:
			print 'invalid option: ' + opt
			sys.exit(1)

	traits = ['trust', 'dom']
	if not os.path.isdir(dir):
		print 'directory does not exist'
		sys.exit(1)
	if not os.path.isdir(dir + 'processed'):
		os.mkdir(dir + 'processed')
	for video in os.listdir(dir):
		if video.endswith('mp4') or video.endswith('avi'):
			print 'begin: ' + dir + 'processed/' + traits[traitID] + '_' + video
			pipeline(
				infile=dir + video,
				outfile=dir + 'processed/' + traits[traitID] + '_' + video[:-3] + 'avi',
				trait=traitID,
				multi=multiFace
			)


if __name__ == '__main__':
	main(sys.argv[1:])

