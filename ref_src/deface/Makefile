# Makefile for fsl_deface

include ${FSLCONFDIR}/default.mk

PROJNAME = fsl_deface

SCRIPTS = fsl_deface fsl_gen_3D

schedule:
	@if [ ! -d ${DESTDIR}/etc ] ; then ${MKDIR} ${DESTDIR}/etc ; ${CHMOD} g+w ${DESTDIR}/etc ; fi
	@if [ ! -d ${DESTDIR}/etc/flirtsch ] ; then ${MKDIR} ${DESTDIR}/etc/flirtsch ; ${CHMOD} g+w ${DESTDIR}/etc/flirtsch ; fi
	${CP} -rf *.mat ${DESTDIR}/etc/flirtsch/.

all:    schedule
