#include "LKH.h"
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>

GainType SolveTSP(int Dimension, char *ParFileName,
                  char *TourFileName, int *Tour, GainType Optimum,
                  GainType Deduction);

enum TourType { INITIAL, INPUT, MERGE, SUBPROBLEM };
static void WriteFullTour(enum TourType Type, int Dimension,
                          char *TourFileName, int Id);

/*
 * The SolveGTSP function solves an E-GTSP instance. The resulting g-tour is
 * returned in the parameter GTour, and its cost as the value of the 
 * function.
 *
 * The algorithm is as follows:
 *. 1. Transform the E-GTSP instance into an asymmetric TSP instance.
 *  2. Write the TSP instance to a problem file.
 *  3. Write suitable parameter values to a parameter file.
 *  4. Execute LKH given these two files (by calling solveTSP).
 *  5. Extract the g-tour from the TSP solution tour by picking the 
 *     first vertex from each cluster in the TSP tour.
 */

GainType SolveGTSP(int *GTour)
{
    int i, j, Dist, Clusters = 0;
    Cluster *Cl;
    Node *From, *To;
    FILE *ParFile, *ProblemFile;
    char ParFileName[256], ProblemFileName[256], TourFileName[256],
        NewInitialTourFileName[256] = { 0 }, 
        NewInputTourFileName[256] = { 0 }, 
        NewSubproblemTourFileName[256] = { 0 },
        **NewMergeTourFileName, 
        Prefix[256];
    GainType M, Cost;
    int *Tour, *Used;

    assert(NewMergeTourFileName =
           (char **) malloc(MergeTourFiles * sizeof(char *)));
    for (i = 0; i < MergeTourFiles; i++)
        assert(NewMergeTourFileName[i] = (char *) malloc(256));
    for (Cl = FirstCluster; Cl; Cl = Cl->Next) {
        Clusters++;
        From = Cl->First;
        do
            From->V = Clusters;
        while ((From = From->Next) != Cl->First);
    }
    assert(Clusters == GTSPSets);

    M = Clusters < 2 ? 0 : INT_MAX / 4 / Precision;

    sprintf(Prefix, "%s.pid%d", Name, getpid());

    /* Create the problem file */
    sprintf(ProblemFileName, "TMP/%s.atsp", Prefix);
    assert(ProblemFile = fopen(ProblemFileName, "w"));
    fprintf(ProblemFile, "NAME : %s.gtsp\n", Prefix);
    fprintf(ProblemFile, "TYPE : ATSP\n");
    if (ProblemType != ATSP)
        fprintf(ProblemFile, "DIMENSION : %d\n", Dimension);
    else
        fprintf(ProblemFile, "DIMENSION : %d\n", DimensionSaved);
    fprintf(ProblemFile, "EDGE_WEIGHT_TYPE : EXPLICIT\n");
    fprintf(ProblemFile, "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n");
    fprintf(ProblemFile, "EDGE_WEIGHT_SECTION\n");

    /* Transform the GTSP into an ATSP */
    for (i = 1; i <= DimensionSaved; i++) {
        From = &NodeSet[i];
        for (j = 1; j <= DimensionSaved; j++) {
            if (i == j)
                fprintf(ProblemFile, "999999 ");
            else {
                To = &NodeSet[j];
                Dist = To == From->Next ? (GainType) 0 :
                    To->V == From->V ? 2 * M :
                    (ProblemType != ATSP ? Distance(From->Next, To) :
                     From->Next->C[j]) + M;
                fprintf(ProblemFile, "%d ", Dist);
                while (Dist * Precision / Precision != Dist) {
                    printff("*** PRECISION (= %d) is too large. ",
                            Precision);
                    if ((Precision /= 10) < 1)
                        Precision = 1;
                    printff("Changed to %d.\n", Precision);
                }
            }
        }
        fprintf(ProblemFile, "\n");
    }
    fprintf(ProblemFile, "EOF\n");
    fclose(ProblemFile);

    /* Create the parameter file */
    sprintf(ParFileName, "TMP/%s.par", Prefix);
    assert(ParFile = fopen(ParFileName, "w"));
    fprintf(ParFile, "PROBLEM_FILE = TMP/%s.atsp\n", Prefix);
    fprintf(ParFile, "ASCENT_CANDIDATES = %d\n", AscentCandidates);
    fprintf(ParFile, "BACKBONE_TRIALS = %d\n", BackboneTrials);
    if (Backtracking)
        fprintf(ParFile, "BACKTRACKING  = YES\n");
    for (i = 0; i < CandidateFiles; i++)
        fprintf(ParFile, "CANDIDATE_FILE = %s\n", CandidateFileName[i]);
    fprintf(ParFile, "CANDIDATE_SET_TYPE = ALPHA\n");
    if (Excess > 0)
        fprintf(ParFile, "EXCESS = %g\n", Excess);
    if (!Gain23Used)
        fprintf(ParFile, "GAIN23 = NO\n");
    if (!GainCriterionUsed)
        fprintf(ParFile, "GAIN_CRITERION = NO\n");
    fprintf(ParFile, "INITIAL_PERIOD = %d\n", InitialPeriod);
    if (InitialTourAlgorithm != WALK)
        fprintf(ParFile, "INITIAL_TOUR_ALGORITHM = %s\n",
                InitialTourAlgorithm ==
                NEAREST_NEIGHBOR ? "NEAREST-NEIGHBOR" :
                InitialTourAlgorithm == GREEDY ? "GREEDY" : "");
    fprintf(ParFile, "INITIAL_STEP_SIZE = %d\n", InitialStepSize);
    if (InitialTourFileName) {
        sprintf(NewInitialTourFileName, "TMP/%s.initial.tour", Prefix);
        WriteFullTour(INITIAL, DimensionSaved, NewInitialTourFileName, 0);
        fprintf(ParFile, "INITIAL_TOUR_FILE = %s\n",
                NewInitialTourFileName);
    }
    fprintf(ParFile, "INITIAL_TOUR_FRACTION = %0.3f\n",
            InitialTourFraction);
    if (InputTourFileName) {
        sprintf(NewInputTourFileName, "TMP/%s.input.tour", Prefix);
        WriteFullTour(INPUT, DimensionSaved, NewInputTourFileName, 0);
        fprintf(ParFile, "INPUT_TOUR_FILE = %s\n", NewInputTourFileName);
    }
    fprintf(ParFile, "KICK_TYPE = %d\n", KickType);
    fprintf(ParFile, "MAX_BREADTH = %d\n", MaxBreadth);
    fprintf(ParFile, "MAX_CANDIDATES = %d%s\n", MaxCandidates,
            CandidateSetSymmetric ? " SYMMETRIC" : "");
    fprintf(ParFile, "MAX_SWAPS = %d\n", MaxSwaps);
    fprintf(ParFile, "MAX_TRIALS = %d\n", MaxTrials);
    for (i = 0; i < MergeTourFiles; i++) {
        sprintf(NewMergeTourFileName[i],
                "TMP/%s.merge%d.tour", Prefix, i + 1);
        WriteFullTour(MERGE, DimensionSaved, NewMergeTourFileName[i], i);
        fprintf(ParFile, "MERGE_TOUR_FILE = %s\n",
                NewMergeTourFileName[i]);
    }
    fprintf(ParFile, "MOVE_TYPE = %d\n", MoveType);
    if (NonsequentialMoveType >= 4)
        fprintf(ParFile, "NONSEQUENTIAL_MOVE_TYPE = %d\n",
                NonsequentialMoveType);
    if (Optimum != MINUS_INFINITY)
        fprintf(ParFile, "OPTIMUM = " GainFormat "\n",
                Optimum + Clusters * M);
    fprintf(ParFile, "PATCHING_A = %d %s\n", PatchingA,
            PatchingARestricted ? "RESTRICTED" :
            PatchingAExtended ? "EXTENDED" : "");
    fprintf(ParFile, "PATCHING_C = %d %s\n", PatchingC,
            PatchingCRestricted ? "RESTRICTED" :
            PatchingCExtended ? "EXTENDED" : "");
    if (PiFileName)
        fprintf(ParFile, "PI_FILE = %s\n", PiFileName);
    fprintf(ParFile, "POPULATION_SIZE = %d\n", MaxPopulationSize);
    fprintf(ParFile, "PRECISION = %d\n", Precision);
    if (!RestrictedSearch)
        fprintf(ParFile, "RESTRICTED_SEARCH = NO\n");
    fprintf(ParFile, "RUNS = %d\n", Runs);
    fprintf(ParFile, "SEED = %d\n", Seed);
    if (!StopAtOptimum)
        fprintf(ParFile, "STOP_AT_OPTIMUM = NO\n");
    if (!Subgradient)
        fprintf(ParFile, "SUBGRADIENT = NO\n");
    if (SubproblemSize > 0)
        fprintf(ParFile, "SUBPROBLEM_SIZE = %d\n", 2 * SubproblemSize);
    if (SubproblemTourFileName) {
        sprintf(NewSubproblemTourFileName, "TMP/%s.subproblem.tour",
                Prefix);
        WriteFullTour(SUBPROBLEM, DimensionSaved, NewSubproblemTourFileName, 0);
        fprintf(ParFile, "SUBPROBLEM_TOUR_FILE = %s\n", 
                NewSubproblemTourFileName);
    }
    fprintf(ParFile, "SUBSEQUENT_MOVE_TYPE = %d\n", SubsequentMoveType);
    if (!SubsequentPatching)
        fprintf(ParFile, "SUBSEQUENT_PATCHING = NO\n");
    if (TimeLimit != DBL_MAX)
        fprintf(ParFile, "TIME_LIMIT = %0.1f\n", TimeLimit);
    sprintf(TourFileName, "TMP/%s.tour", Prefix);
    fprintf(ParFile, "TOUR_FILE = %s\n", TourFileName);
    fprintf(ParFile, "TRACE_LEVEL = %d\n",
            TraceLevel == 0 ? 1 : TraceLevel);
    fclose(ParFile);

    /* Solve the ATSP */
    assert(Tour = (int *) malloc((DimensionSaved + 1) * sizeof(int)));
    Cost =
        SolveTSP(DimensionSaved, ParFileName, TourFileName,
                 Tour, Optimum, Clusters * M);
    unlink(ParFileName);
    unlink(ProblemFileName);
    unlink(NewInitialTourFileName);
    unlink(NewInputTourFileName);
    for (i = 0; i < MergeTourFiles; i++)
        unlink(NewMergeTourFileName[i]);
    unlink(NewSubproblemTourFileName);

    /* Extract the g-tour and check it */
    for (i = 1; i <= DimensionSaved; i++)
        NodeSet[Tour[i - 1]].Suc = &NodeSet[Tour[i]];
    free(Tour);
    From = FirstNode;
    i = From->V;
    do
        FirstNode = From = From->Suc;
    while (From->V == i);
    assert(Used = (int *) calloc(Clusters + 1, sizeof(int)));
    i = 0;
    do {
        GTour[++i] = From->Id;
        j = From->V;
        if (Used[j])
            eprintf("Illegal g-tour: cluster entered more than once");
        Used[j] = 1;
        while (From->V == j)
            From = From->Suc;
    } while (From != FirstNode);
    free(Used);
    if (i != Clusters)
        eprintf("Illegal g-tour: unvisited cluster(s)");
    GTour[0] = GTour[Clusters];
    return Cost;
}

static void WriteFullTour(enum TourType Type, int Dimension,
                         char *TourFileName, int Id)
{
    int *Tour, i = 0;
    Node *N, *From, *To;

    assert(Tour = (int *) malloc((Dimension + 1) * sizeof(int)));
    From = FirstNode;
    while (Type == INITIAL ? !From->InitialSuc :
           Type == INPUT ? !From->InputSuc :
           Type == MERGE ? !From->MergeSuc[Id] :
           Type == SUBPROBLEM ? !From->SubproblemSuc : 0)
        From = From->Suc;
    N = From;
    do {
        To = N;
        do
            Tour[++i] = N->Id;
        while ((N = N->Next) != To);
        N = Type == INITIAL ? N->InitialSuc :
            Type == INPUT ? N->InputSuc :
            Type == MERGE ? N->MergeSuc[Id] :
            Type == SUBPROBLEM ? N->SubproblemSuc : From;
    } while (N != From);
    assert(i == Dimension);
    Tour[0] = Tour[Dimension];
    WriteTour(TourFileName, Tour, -1);
    free(Tour);
}
