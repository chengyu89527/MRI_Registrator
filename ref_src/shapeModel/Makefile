# $Id$
include ${FSLCONFDIR}/default.mk

PROJNAME = shapeModel

SHAPEOBJS=shapeModel.o
USRINCFLAGS = -I${INC_NEWMAT} -I${INC_ZLIB}
USRLDFLAGS = -L${LIB_NEWMAT}  -L${LIB_ZLIB}
 

all: shapeModel.a 

shapeModel.a: ${SHAPEOBJS}
	${AR} -r libshapeModel.a ${SHAPEOBJS}




