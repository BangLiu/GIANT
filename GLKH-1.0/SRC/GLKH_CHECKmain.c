#include "LKH.h"

int main(int argc, char *argv[])
{
    GainType Cost;
    Cluster *Cl;
    Node *From;
    int *GTour, *Used, Clusters = 0, i;
    FILE *GTourFile;
    char *GTourFileName, *Line, *Keyword, Buf1[256], Buf2[256];
    char Delimiters[] = " :=\n\t\r\f\v\xef\xbb\xbf";

    /* Read the specification of the problem */

    sprintf(Buf1, "GTSPLIB/%s.gtsp", argv[1]);
    ProblemFileName = Buf1;
    sprintf(Buf2, "G-TOURS/%s.%s.tour", argv[1], argv[2]);
    GTourFileName = Buf2;
    MaxMatrixDimension = 0;
    ReadProblem();
    for (Cl = FirstCluster; Cl; Cl = Cl->Next) {
        Clusters++;
        From = Cl->First;
        do
            From->V = Clusters;
        while ((From = From->Next) != Cl->First);
    }
    if (!(GTourFile = fopen(GTourFileName, "r")))
        eprintf("Cannot find %s\n", GTourFileName);
    while ((Line = ReadLine(GTourFile))) {
        if (!(Keyword = strtok(Line, Delimiters)))
            continue;
        for (i = 0; i < strlen(Keyword); i++)
            Keyword[i] = (char) toupper(Keyword[i]);
        if (!strcmp(Keyword, "TOUR_SECTION"))
            break;
        if (Optimum == 0 && !strcmp(Keyword, "COMMENT")) {
            Keyword = strtok(0, Delimiters);
            Keyword = strtok(0, Delimiters);
            Optimum = atoi(Keyword);
        } else {
            if (!strcmp(Keyword, "DIMENSION")) {
                int n;
                Keyword = strtok(0, Delimiters);
                n = atoi(Keyword);
                if (n != Clusters)
                    eprintf("Wrong number of clusters");
            }
        }
    }
    assert(GTour = (int *) malloc((Clusters + 1) * sizeof(int)));
    for (i = 1; i <= Clusters; i++)
        fscanf(GTourFile, "%d", &GTour[i]);
    GTour[0] = GTour[Clusters];
    assert(Used = (int *) calloc(Clusters + 1, sizeof(int)));
    Cost = 0;
    for (i = 1; i <= Clusters; i++) {
        Cost += ProblemType != ATSP ?
            Distance(&NodeSet[GTour[i - 1]], &NodeSet[GTour[i]]) :
            NodeSet[GTour[i - 1]].C[GTour[i]];
        if (Used[NodeSet[GTour[i]].V])
            eprintf("Cluster entered more than once");
        Used[NodeSet[GTour[i]].V] = 1;
    }
    for (i = 1; i <= Clusters; i++)
        if (!Used[i])
            eprintf("Unvisited cluster(s)");
    if (Cost != Optimum)
        eprintf("Cost = %lld != Optimum = %lld\n", Cost, Optimum);
    printff("OK. Cost = %lld\n", Cost);
    free(Used);
    return 0;
}
