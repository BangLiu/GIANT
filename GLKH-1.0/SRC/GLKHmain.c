#include "LKH.h"
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

GainType SolveGTSP(int *GTour);
GainType PostOptimize(int *GTour, GainType Cost);
static double GetTimeUsage(int who);

/*
 * This file contains the main function of the GLKH program.
 */

int main(int argc, char *argv[])
{
    int *GTour;
    GainType Cost;

    /* Read the specification of the problem */
    if (argc >= 2)
        ParameterFileName = argv[1];
    ReadParameters();
    ReadProblem();

    assert(GTour = (int *) malloc((GTSPSets + 1) * sizeof(int)));
    Cost = SolveGTSP(GTour);
    Cost = PostOptimize(GTour, Cost);
    if (OutputTourFileName) {
        TraceLevel = 1;
        Dimension = ProblemType != ATSP ? GTSPSets : 2 * GTSPSets;
        WriteTour(OutputTourFileName, GTour, Cost);
    }
    if (TourFileName && (Optimum == MINUS_INFINITY || Cost < Optimum)) {
        TraceLevel = 1;
        Dimension = ProblemType != ATSP ? GTSPSets : 2 * GTSPSets;
        WriteTour(TourFileName, GTour, Cost);
    }
    printff("Value = " GainFormat, Cost);
    if (Optimum != MINUS_INFINITY && Optimum != 0)
        printff(", Error = %0.2f%%", 100.0 * (Cost - Optimum) / Optimum);
    printff(", Time = %0.1f sec.\n\n",
            GetTimeUsage(RUSAGE_SELF) + GetTimeUsage(RUSAGE_CHILDREN));
    return 0;
}

static double GetTimeUsage(int who)
{
    struct rusage ru;
    getrusage(who, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1000000.0;
}
