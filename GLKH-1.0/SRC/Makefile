# CC = gcc
IDIR = INCLUDE
ODIR = OBJ
CFLAGS = -O3 -Wall -I$(IDIR) -g

_DEPS = GainType.h LKH.h

DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ = SolveGTSP.o SolveTSP.o PostOptimize.o                           \
       Distance.o Distance_SPECIAL.o eprintf.o fscanint.o GetTime.o    \
       IsCandidate.o printff.o PrintParameters.o Random.o ReadLine.o   \
	   ReadParameters.o ReadPenalties.o ReadProblem.o Statistics.o     \
	   WriteTour.o 

_OBJ_GLKH = GLKHmain.o $(_OBJ)
_OBJ_GLKH_EXP = GLKH_EXPmain.o $(_OBJ)
_OBJ_GLKH_CHECK = GLKH_CHECKmain.o $(_OBJ)
             
OBJ_GLKH = $(patsubst %,$(ODIR)/%,$(_OBJ_GLKH))
OBJ_GLKH_EXP = $(patsubst %,$(ODIR)/%,$(_OBJ_GLKH_EXP))
OBJ_GLKH_CHECK= $(patsubst %,$(ODIR)/%,$(_OBJ_GLKH_CHECK))

$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: 
	all clean

all:
	$(MAKE) GLKH
	$(MAKE) GLKH_EXP
	$(MAKE) GLKH_CHECK

GLKH: $(OBJ_GLKH) $(DEPS)
	$(CC) -o ../GLKH $(OBJ_GLKH) $(CFLAGS) -lm
GLKH_EXP: $(OBJ_GLKH_EXP) $(DEPS)
	$(CC) -o ../GLKH_EXP $(OBJ_GLKH_EXP) $(CFLAGS) -lm
GLKH_CHECK: $(OBJ_GLKH_CHECK) $(DEPS)
	$(CC) -o ../GLKH_CHECK $(OBJ_GLKH_CHECK) $(CFLAGS) -lm
 
clean:
	/bin/rm -f $(ODIR)/*.o ../GLKH ../GLKH_EXP ../GLKH_CHECK *~ ._*\
   	           $(IDIR)/*~ $(IDIR)/._* 
