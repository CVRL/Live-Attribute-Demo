"""
Main file for a continuous live demo from the webcam
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


def keyPress(key=None):
	if key == 27:
		resetFiles()
		cv2.destroyAllWindows()
		return True


def pipeline(trait=0, scale=1, loc=(0,0), multi=False):
	"""
	trait: integer - 0:Trustworthiness, 1:Dominance
	scale: Scale size of final window
	loc: location (x, y) of final window on screen
	multi:  False: Analyze one face over time,
			True: Analyze multiple faces by frame
	"""
	cap = cv2.VideoCapture(0)
	graphic = Graphics()

	while cap.isOpened():
		_, frame = cap.read()
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
		graphic.displayFrame(display, scale, loc)
		if keyPress(cv2.waitKey(30)):
			break

def main(argv):
	"""process arguments"""
	def usage():
		print '\npython processLive.py <args>'
		print '\t--trait=[0, 1]'
		print '\t--multi'
		print '\t--scale=[float > 0]'
		print '\t--x=[integer]'
		print '\t--y=[integer]'
		print '\t--help\n'

	try:
		opts, args = getopt.getopt(argv, 't:s:x:y:mh', ['trait=', 'scale=', 'x=', 'y=', 'multi', 'help'])
	except getopt.GetoptError:
		print 'incorrect argument'
		sys.exit(1)

	defaults = (0, 1, [0,0], False)
	traitID, scaleFactor, screenLoc, multiFace = defaults
	for opt, arg in opts:
		if opt in ['-t', '--trait']:
			traitID = int(arg)
		elif opt in ['-m', '--multi']:
			multiFace = 1
		elif opt in ['-s', '--scale']:
			scaleFactor = float(arg)
		elif opt in ['-x', '--x']:
			screenLoc[0] = int(arg)
		elif opt in ['-y', '--y']:
			screenLoc[1] = int(arg)
		elif opt in ['-h', '--help']:
			usage()
			sys.exit(0)
		else:
			print 'invalid option: ' + opt
			sys.exit(1)

	print 'entering pipeline'
	pipeline(traitID, scaleFactor, screenLoc, multiFace)


if __name__ == '__main__':
	main(sys.argv[1:])



