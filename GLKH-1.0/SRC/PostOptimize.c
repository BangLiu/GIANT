#include "LKH.h"
#include <unistd.h>

GainType SolveTSP(int Dimension, char *ParFileName,
                  char *TourFileName, int *Tour, GainType Optimum,
                  GainType Displacement);

static GainType KOptimize(int *GTour);
static GainType ClusterOptimize(int *GTour);

/*
 * The PostOptimize function performs post optimization on a given g-tour
 * by two means: (1) Using LKH for local optimization on an instance with 
 * the vertices of the g-tour. (2) Performing so-called cluster optimization.
 * This heuristic attempts to find a g-tour that visits the clusters in the 
 * same order as the current g-tour, but is cheaper than this
 *
 * Local optimization and cluster optimization are performed as long as it is
 * possible to improve the current best g-tour.
 *
 * Parameters
 *   GTour: The input g-tour
 *   Cost:  The cost of GTour
 *
 * The resulting g-tour is returned in the parameter GTour, and its cost 
 * as the value of the function.
 */

GainType PostOptimize(int *GTour, GainType Cost)
{
    int *BestGTour, i, Clusters = GTSPSets, FirstTime = 1;

    if (StopAtOptimum && Cost == Optimum)
        return Cost;
    assert(BestGTour = (int *) malloc((Clusters + 1) * sizeof(int)));
    memcpy(BestGTour, GTour, (Clusters + 1) * sizeof(int));
    BestCost = Cost;
    if (TraceLevel > 0)
        printff("Post-optimization [Cost = " GainFormat "]\n", Cost);

    while (1) {
        /* Perform K-opt optimization */
        if (Clusters >= 4)
            Cost = KOptimize(GTour);
        if (Cost < BestCost) {
            for (i = 1; i <= Clusters; i++)
                GTour[i] = BestGTour[GTour[i]];
            GTour[0] = GTour[Clusters];
            memcpy(BestGTour, GTour, (Clusters + 1) * sizeof(int));
            BestCost = Cost;
            if (TraceLevel > 0)
                printff("                  K-opt = " GainFormat "\n",
                        Cost);
        } else {
            memcpy(GTour, BestGTour, (Clusters + 1) * sizeof(int));
            if (!FirstTime)
                break;
        }
        if (StopAtOptimum && BestCost == Optimum)
            break;
        /* Perform cluster optimization */
        Cost = ClusterOptimize(GTour);
        if (Cost >= BestCost)
            break;
        memcpy(BestGTour, GTour, (Clusters + 1) * sizeof(int));
        BestCost = Cost;
        FirstTime = 0;
        if (TraceLevel > 0)
            printff("                  C-opt = " GainFormat "\n", Cost);
        if (StopAtOptimum && BestCost == Optimum)
            break;
    }
    memcpy(GTour, BestGTour, (Clusters + 1) * sizeof(int));
    free(BestGTour);
    return BestCost;
}

/*
 * The KOptimize function attempts to improve a given g-tour, GTour, using LKH.
 * The return value is the cost of the resulting g-tour.
 */

static GainType KOptimize(int *GTour)
{
    int i, j, Clusters = GTSPSets;
    Node *N;
    FILE *ParFile, *ProblemFile;
    char ParFileName[256], ProblemFileName[256], TourFileName[256],
        Prefix[256];
    GainType Cost;

    /* Create the problem file */
    sprintf(Prefix, "%s.pid%d", Name, getpid());
    sprintf(ProblemFileName, "TMP/%s.post.tsp", Prefix);
    assert(ProblemFile = fopen(ProblemFileName, "w"));
    fprintf(ProblemFile, "NAME : %s\n", Name);
    fprintf(ProblemFile, "TYPE : %s\n", Type);
    fprintf(ProblemFile, "DIMENSION : %d\n", Clusters);
    fprintf(ProblemFile, "EDGE_WEIGHT_TYPE : %s\n", EdgeWeightType);
    if (CoordType == NO_COORDS) {
        fprintf(ProblemFile, "EDGE_WEIGHT_FORMAT : FULL_MATRIX\n");
        fprintf(ProblemFile, "EDGE_WEIGHT_SECTION\n");
        for (i = 1; i <= Clusters; i++) {
            for (j = 1; j <= Clusters; j++)
                fprintf(ProblemFile, "%d ", i == j ? 99999 :
                        ProblemType != ATSP ? Distance(&NodeSet[GTour[i]],
                                                       &NodeSet[GTour[j]])
                        : NodeSet[GTour[i]].C[GTour[j]]);
            fprintf(ProblemFile, "\n");
        }
    } else {
        fprintf(ProblemFile, "NODE_COORD_SECTION\n");
        if (CoordType == TWOD_COORDS) {
            for (i = 1; i <= Clusters; i++) {
                N = &NodeSet[GTour[i]];
                fprintf(ProblemFile, "%d %f %f\n", i, N->X, N->Y);
            }
        } else if (CoordType == THREED_COORDS) {
            for (i = 1; i <= Clusters; i++) {
                N = &NodeSet[GTour[i]];
                fprintf(ProblemFile, "%d %f %f %f\n", i, N->X, N->Y, N->Z);
            }
        }
    }
    fprintf(ProblemFile, "EOF\n");
    fclose(ProblemFile);

    /* Create the parameter file */
    sprintf(ParFileName, "TMP/%s.post.par", Prefix);
    assert(ParFile = fopen(ParFileName, "w"));
    fprintf(ParFile, "PROBLEM_FILE = %s\n", ProblemFileName);
    fprintf(ParFile, "MAX_TRIALS = %d\n", MaxTrials);
    fprintf(ParFile, "OPTIMUM = " GainFormat "\n", BestCost);
    fprintf(ParFile, "PRECISION = %d\n", Precision);
    fprintf(ParFile, "RUNS = 1\n");
    fprintf(ParFile, "SEED = %d\n", Seed);
    if (!Subgradient)
        fprintf(ParFile, "SUBGRADIENT = NO\n");
    sprintf(TourFileName, "TMP/%s.post.tour", Prefix);
    fprintf(ParFile, "TOUR_FILE = %s\n", TourFileName);
    fclose(ParFile);

    /* Solve the problem */
    Cost =
        SolveTSP(Clusters, ParFileName, TourFileName, GTour, BestCost, 0);
    unlink(ParFileName);
    unlink(ProblemFileName);
    return Cost;
}

/*
 * The ClusterOptimize function attempts to improve a given g-tour, GTour, 
 * using cluster optimization (shortest path in a layered network).
 * See M. Fischetti et al.: A branch-and-cut algorithm for the symmetric 
 * generalized traveling salesman problem.
 * Operations Research, 45:378-394, 1997.
 *
 * The return value is the cost of the resulting g-tour.
 */

static GainType ClusterOptimize(int *GTour)
{
    int i, j = 1, d, MinSize, MinV, Clusters = GTSPSets;
    Cluster *Cl;
    Node **First, *From, *To;
    GainType Cost = PLUS_INFINITY;

    MinSize = MinV = INT_MAX;
    for (Cl = FirstCluster; Cl && MinSize != 1; Cl = Cl->Next) {
        if (Cl->Size < MinSize) {
            MinSize = Cl->Size;
            MinV = Cl->First->V;
        }
    }
    for (i = 1; i <= Clusters; i++) {
        From = &NodeSet[GTour[i]];
        if (From->V == MinV) {
            j = i;
            break;
        }
    }
    assert(First = (Node **) malloc(Clusters * sizeof(Node *)));
    for (i = 0; i < Clusters; i++) {
        First[i] = &NodeSet[GTour[j]];
        if (++j > Clusters)
            j = 1;
    }
    FirstNode = First[0];
    do {
        FirstNode->Cost = 0;
        FirstNode->Dad = 0;
        To = First[1];
        do {
            To->Cost = ProblemType != ATSP ? Distance(FirstNode, To)
                : FirstNode->C[To->Id];
            To->Dad = FirstNode;
        } while ((To = To->Next) != First[1]);
        for (i = 2; i < Clusters; i++) {
            To = First[i];
            do {
                To->Cost = INT_MAX;
                From = First[i - 1];
                do {
                    if ((d = From->Cost +
                         (ProblemType != ATSP ? Distance(From, To) :
                          From->C[To->Id])) < To->Cost) {
                        To->Dad = From;
                        To->Cost = d;
                    }
                } while ((From = From->Next) != First[i - 1]);
            } while ((To = To->Next) != First[i]);
        }
        From = First[Clusters - 1];
        do {
            if ((d = From->Cost +
                 (ProblemType != ATSP ? Distance(From, FirstNode) :
                  From->C[FirstNode->Id])) < Cost) {
                Cost = d;
                for (i = Clusters, To = From; To; To = To->Dad)
                    GTour[i--] = To->Id;
            }
        } while ((From = From->Next) != First[Clusters - 1]);
    } while ((FirstNode = FirstNode->Next) != First[0]);
    free(First);
    GTour[0] = GTour[Clusters];
    return Cost;
}
