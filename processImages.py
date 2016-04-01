"""
Main file for processing all images in a given directory
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


def pipeline(infile, outfile, trait=0):
	"""
	infile: input image
	outfile: output image name
	trait: integer - 0:Trustworthiness, 1:Dominance
	"""
	graphic = Graphics()
	frame = cv2.imread(infile)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face = None

	faceCoordsList = findFaces(gray)
	faceList = []
	if faceCoordsList is not None:
		for faceCoords in faceCoordsList:
			face = Face(gray, faceCoords, trait)
			if face.predictValue():
				faceList.append(face)
			face = None
			resetFiles()

	display = graphic.draw(frame, faceList, trait, True)
	cv2.imwrite(outfile, display)

def main(argv):
	"""process arguments"""

	def usage():
		print 'processes all jpg files in a given directory'
		print '\nUsage: python processLive.py <args>'
		print '\t--trait=[0, 1]'
		print '\t--dir=[string]'
		print '\t--help\n'

	try:
		opts, args = getopt.getopt(argv, 't:d:h', ['trait=', 'dir=', 'multi', 'help'])
	except getopt.GetoptError:
		print 'incorrect argument'
		sys.exit(1)

	defaults = (0, None)
	traitID, dir = defaults
	for opt, arg in opts:
		if opt in ['-t', '--trait']:
			traitID = int(arg)
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
	for image in os.listdir(dir):
		if image.endswith('jpg'):
			print 'begin: ' + dir + 'processed/' + traits[traitID] + '_' + image
			pipeline(
				infile=dir + image,
				outfile=dir + 'processed/' + traits[traitID] + '_' + image,
				trait=traitID,
			)

if __name__ == '__main__':
	main(sys.argv[1:])

