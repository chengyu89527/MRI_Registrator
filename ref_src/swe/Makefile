include ${FSLCONFDIR}/default.mk

PROJNAME = swe

USRINCFLAGS = -I${INC_NEWMAT} -I${INC_PROB} -I${INC_ZLIB}
USRLDFLAGS = -L${LIB_NEWMAT} -L${LIB_PROB} -L${LIB_ZLIB}

LIBS = -lnewimage -lmiscmaths -lutils -lm -lnewmat -lNewNifti -lznz -lprob -lz

XFILES = swe

SCRIPTS = swe

OBJS = swe.o sweopts.o swefns.o

all: libswe.a ${XFILES}

libswe.a: ${OBJS}
	${AR} -r libswe.a ${OBJS}

swe:  ${OBJS} 
	    ${CXX}  ${CXXFLAGS} ${LDFLAGS} -o $@ ${OBJS}  ${LIBS}
