/* generated automatically from cddio.c */
/* cddio.c:  Basic Input and Output Procedures for cddlib
   written by Komei Fukuda, fukuda@math.ethz.ch
*/

/* cddlib : C-library of the double description method for
   computing all vertices and extreme rays of the polyhedron 
   P= {x :  b - A x >= 0}.  
   Please read COPYING (GNU General Public Licence) and
   the manual cddlibman.tex for detail.
*/

#include "setoper.h"
#include "cdd_f.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

/* void ddf_fread_rational_value (FILE *, myfloat *); */
void ddf_SetLinearity(ddf_MatrixPtr, char *);

void ddf_SetInputFile(FILE **f,ddf_DataFileType inputfile,ddf_ErrorType *Error)
{
  int opened=0,stop,quit=0;
  int i,dotpos=0,trial=0;
  char ch;
  char *tempname;
  
  
  *Error=ddf_NoError;
  while (!opened && !quit) {
    fprintf(stderr,"\n>> Input file: ");
    scanf("%s",inputfile);
    ch=getchar();
    stop=ddf_FALSE;
    for (i=0; i<ddf_filenamelen && !stop; i++){
      ch=inputfile[i];
      switch (ch) {
        case '.': 
          dotpos=i+1;
          break;
        case ';':  case ' ':  case '\0':  case '\n':  case '\t':     
          stop=ddf_TRUE;
          tempname=(char*)calloc(ddf_filenamelen,sizeof(ch));
          strncpy(tempname,inputfile,i);
          strcpy(inputfile,tempname);
          free(tempname);
          break;
      }
    }
    if ( ( *f = fopen(inputfile,"r") )!= NULL) {
      fprintf(stderr,"input file %s is open\n",inputfile);
      opened=1;
      *Error=ddf_NoError;
    }
    else{
      fprintf(stderr,"The file %s not found\n",inputfile);
      trial++;
      if (trial>5) {
        *Error=ddf_IFileNotFound;
        quit=1;
      }
    }
  }
}

void ddf_SetWriteFileName(ddf_DataFileType inputfile, ddf_DataFileType outfile, char cflag, ddf_RepresentationType rep)
{
  char *extension;
  ddf_DataFileType ifilehead="";
  int i,dotpos;
  
  switch (cflag) {
    case 'o':
      switch (rep) {
        case ddf_Generator:
          extension=".ine"; break;     /* output file for ine data */
        case ddf_Inequality:
          extension=".ext"; break;     /* output file for ext data */
        default: 
          extension=".xxx";break;
      }
      break;

    case 'a':         /* decide for output adjacence */
      if (rep==ddf_Inequality)
        extension=".ead";       /* adjacency file for ext data */
      else
        extension=".iad";       /* adjacency file for ine data */
      break;
    case 'i':         /* decide for output incidence */
      if (rep==ddf_Inequality)
        extension=".ecd";       /* ext incidence file */
      else
        extension=".icd";       /* ine incidence file */
      break;
    case 'n':         /* decide for input incidence */
      if (rep==ddf_Inequality)
        extension=".icd";       /* ine incidence file */
      else
        extension=".ecd";       /* ext incidence file */
      break;
    case 'j':        /* decide for input adjacency */
      if (rep==ddf_Inequality)
        extension=".iad";       /* ine adjacency file */
      else
        extension=".ead";       /* ext adjacency file */
      break;
    case 'l':
      extension=".ddl";break;   /* log file */
    case 'd':
      extension=".dex";break;   /* decomposition output */
    case 'p':
      extension="sub.ine";break;  /* preprojection sub inequality file */
    case 'v':
      extension=".solved";break;  /* verify_input file */
    case 's':
      extension=".lps";break;   /* LP solution file */
    default:
      extension=".xxx";break;
  }
  dotpos=-1;
  for (i=0; i< strlen(inputfile); i++){
    if (inputfile[i]=='.') dotpos=i;
  }
  if (dotpos>1) strncpy(ifilehead, inputfile, dotpos);
  else strcpy(ifilehead,inputfile);
  if (strlen(inputfile)<=0) strcpy(ifilehead,"tempcdd");
  strcpy(outfile,ifilehead); 
  strcat(outfile,extension); 
  if (strcmp(inputfile, outfile)==0) {
    strcpy(outfile,inputfile); 
    strcat(outfile,extension); 
  }
/*  fprintf(stderr,"outfile name = %s\n",outfile);  */
}


ddf_NumberType ddf_GetNumberType(const char *line)
{
  ddf_NumberType nt;

  if (strncmp(line, "integer", 7)==0) {
    nt = ddf_Integer;
  }
  else if (strncmp(line, "rational", 8)==0) {
    nt = ddf_Rational;
  }
  else if (strncmp(line, "real", 4)==0) {
    nt = ddf_Real;
  }
  else { 
    nt=ddf_Unknown;
  }
  return nt;
}

void ddf_ProcessCommandLine(FILE *f, ddf_MatrixPtr M, const char *line)
{
  char newline[ddf_linelenmax];
  ddf_colrange j;
  myfloat value;

  ddf_init(value);
  if (strncmp(line, "hull", 4)==0) {
    M->representation = ddf_Generator;
  }
  if (strncmp(line, "debug", 5)==0) {
    ddf_debug = ddf_TRUE;
#ifdef ddf_GMPRATIONAL
    ddf_debug = ddf_TRUE;
#endif
  }
  if (strncmp(line, "partial_enum", 12)==0 ||
       strncmp(line, "equality", 8)==0  ||
       strncmp(line, "linearity", 9)==0 ) {
    fgets(newline,ddf_linelenmax,f);    
    ddf_SetLinearity(M,newline);
  }
  if (strncmp(line, "maximize", 8)==0 ||
      strncmp(line, "minimize", 8)==0) {
    if (strncmp(line, "maximize", 8)==0) M->objective=ddf_LPmax;
    else M->objective=ddf_LPmin;
    for (j = 1; j <= M->colsize; j++) {
    if (M->numbtype==ddf_Real) {
#if !defined(ddf_GMPRATIONAL)
        double rvalue;
        fscanf(f, "%lf", &rvalue);
        ddf_set_d(value, rvalue);
#endif
      } else {
        ddf_fread_rational_value (f, value);
      }
      ddf_set(M->rowvec[j - 1],value);
      if (ddf_debug) {fprintf(stderr,"cost(%5ld) =",j); ddf_WriteNumber(stderr,value);}
    }  /*of j*/
  }
  ddf_clear(value);
}

ddf_boolean ddf_AppendMatrix2Poly(ddf_PolyhedraPtr *poly, ddf_MatrixPtr M)
{
  ddf_boolean success=ddf_FALSE;
  ddf_MatrixPtr Mpoly,Mnew=NULL;
  ddf_ErrorType err;

  if ((*poly)!=NULL && (*poly)->m >=0 && (*poly)->d>=0 &&
      (*poly)->d==M->colsize && M->rowsize>0){
    Mpoly=ddf_CopyInput(*poly);
    Mnew=ddf_AppendMatrix(Mpoly, M);
    ddf_FreePolyhedra(*poly);
    *poly=ddf_DDMatrix2Poly(Mnew,&err);
    ddf_FreeMatrix(Mpoly);
    ddf_FreeMatrix(Mnew);
    if (err==ddf_NoError) success=ddf_TRUE;
  }
  return success;
}

ddf_MatrixPtr ddf_MatrixCopy(ddf_MatrixPtr M)
{
  ddf_MatrixPtr Mcopy=NULL;
  ddf_rowrange m;
  ddf_colrange d;

  m= M->rowsize;
  d= M->colsize;
  if (m >=0 && d >=0){
    Mcopy=ddf_CreateMatrix(m, d);
    ddf_CopyAmatrix(Mcopy->matrix, M->matrix, m, d);
    ddf_CopyArow(Mcopy->rowvec, M->rowvec, d);
    set_copy(Mcopy->linset,M->linset);
    Mcopy->numbtype=M->numbtype;
    Mcopy->representation=M->representation;
    Mcopy->objective=M->objective;
  }
  return Mcopy;
}

ddf_MatrixPtr ddf_CopyMatrix(ddf_MatrixPtr M)
{
  return ddf_MatrixCopy(M);
}

ddf_MatrixPtr ddf_MatrixNormalizedCopy(ddf_MatrixPtr M)
{
  ddf_MatrixPtr Mcopy=NULL;
  ddf_rowrange m;
  ddf_colrange d;

  m= M->rowsize;
  d= M->colsize;
  if (m >=0 && d >=0){
    Mcopy=ddf_CreateMatrix(m, d);
    ddf_CopyNormalizedAmatrix(Mcopy->matrix, M->matrix, m, d);
    ddf_CopyArow(Mcopy->rowvec, M->rowvec, d);
    set_copy(Mcopy->linset,M->linset);
    Mcopy->numbtype=M->numbtype;
    Mcopy->representation=M->representation;
    Mcopy->objective=M->objective;
  }
  return Mcopy;
}


ddf_MatrixPtr ddf_MatrixAppend(ddf_MatrixPtr M1, ddf_MatrixPtr M2)
{
  ddf_MatrixPtr M=NULL;
  ddf_rowrange i, m,m1,m2;
  ddf_colrange j, d,d1,d2;
  
  m1=M1->rowsize;
  d1=M1->colsize;
  m2=M2->rowsize;
  d2=M2->colsize;

  m=m1+m2;
  d=d1;

  if (d1>=0 && d1==d2 && m1>=0 && m2>=0){
    M=ddf_CreateMatrix(m, d);
    ddf_CopyAmatrix(M->matrix, M1->matrix, m1, d);
    ddf_CopyArow(M->rowvec, M1->rowvec, d);
    for (i=0; i<m1; i++){
      if (set_member(i+1,M1->linset)) set_addelem(M->linset,i+1);
    }
    for (i=0; i<m2; i++){
       for (j=0; j<d; j++)
         ddf_set(M->matrix[m1+i][j],M2->matrix[i][j]);
         /* append the second matrix */
       if (set_member(i+1,M2->linset)) set_addelem(M->linset,m1+i+1);
    }
    M->numbtype=M1->numbtype;
    M->representation=M1->representation;
    M->objective=M1->objective;
  }
  return M;
}

ddf_MatrixPtr ddf_MatrixNormalizedSortedCopy(ddf_MatrixPtr M,ddf_rowindex *newpos)  /* 094 */
{ 
  /* Sort the rows of Amatrix lexicographically, and return a link to this sorted copy. 
  The vector newpos is allocated, where newpos[i] returns the new row index
  of the original row i (i=1,...,M->rowsize). */
  ddf_MatrixPtr Mcopy=NULL,Mnorm=NULL;
  ddf_rowrange m,i;
  ddf_colrange d;
  ddf_rowindex roworder;

  /* if (newpos!=NULL) free(newpos); */
  m= M->rowsize;
  d= M->colsize;
  roworder=(long *)calloc(m+1,sizeof(long));
  *newpos=(long *)calloc(m+1,sizeof(long));
  if (m >=0 && d >=0){
    Mnorm=ddf_MatrixNormalizedCopy(M);
    Mcopy=ddf_CreateMatrix(m, d);
    for(i=1; i<=m; i++) roworder[i]=i;
    
    ddf_RandomPermutation(roworder, m, 123);
    ddf_QuickSort(roworder,1,m,Mnorm->matrix,d);

    ddf_PermuteCopyAmatrix(Mcopy->matrix, Mnorm->matrix, m, d, roworder);
    ddf_CopyArow(Mcopy->rowvec, M->rowvec, d);
    for(i=1; i<=m; i++) {
      if (set_member(roworder[i],M->linset)) set_addelem(Mcopy->linset, i);
      (*newpos)[roworder[i]]=i;
    }
    Mcopy->numbtype=M->numbtype;
    Mcopy->representation=M->representation;
    Mcopy->objective=M->objective;
    ddf_FreeMatrix(Mnorm);
  }
  free(roworder);
  return Mcopy;
}

ddf_MatrixPtr ddf_MatrixUniqueCopy(ddf_MatrixPtr M,ddf_rowindex *newpos)
{ 
  /* Remove row duplicates, and return a link to this sorted copy. 
     Linearity rows have priority over the other rows.
     It is better to call this after sorting with ddf_MatrixNormalizedSortedCopy.
     The vector newpos is allocated, where *newpos[i] returns the new row index
     of the original row i (i=1,...,M->rowsize).  *newpos[i] is negative if the original
     row is dominated by -*newpos[i] and eliminated in the new copy.
  */
  ddf_MatrixPtr Mcopy=NULL;
  ddf_rowrange m,i,uniqrows;
  ddf_rowset preferredrows;
  ddf_colrange d;
  ddf_rowindex roworder;

  /* if (newpos!=NULL) free(newpos); */
  m= M->rowsize;
  d= M->colsize;
  preferredrows=M->linset;
  roworder=(long *)calloc(m+1,sizeof(long));
  if (m >=0 && d >=0){
    for(i=1; i<=m; i++) roworder[i]=i;
    ddf_UniqueRows(roworder, 1, m, M->matrix, d,preferredrows, &uniqrows);

    Mcopy=ddf_CreateMatrix(uniqrows, d);
    ddf_PermutePartialCopyAmatrix(Mcopy->matrix, M->matrix, m, d, roworder,1,m);
    ddf_CopyArow(Mcopy->rowvec, M->rowvec, d);
    for(i=1; i<=m; i++) {
      if (roworder[i]>0 && set_member(i,M->linset)) set_addelem(Mcopy->linset, roworder[i]);
    }
    Mcopy->numbtype=M->numbtype;
    Mcopy->representation=M->representation;
    Mcopy->objective=M->objective;
  }
  *newpos=roworder;
  return Mcopy;
}


ddf_MatrixPtr ddf_MatrixNormalizedSortedUniqueCopy(ddf_MatrixPtr M,ddf_rowindex *newpos) /* 094 */
{ 
  /* Sort and remove row duplicates, and return a link to this sorted copy. 
     Linearity rows have priority over the other rows.
     It is better to call this after sorting with ddf_MatrixNormalizedSortedCopy.
     The vector newpos is allocated, where *newpos[i] returns the new row index
     of the original row i (i=1,...,M->rowsize).  *newpos[i] is negative if the original
     row is dominated by -*newpos[i] and eliminated in the new copy.
  */
  ddf_MatrixPtr M1=NULL,M2=NULL;
  ddf_rowrange m,i;
  ddf_colrange d;
  ddf_rowindex newpos1=NULL,newpos1r=NULL,newpos2=NULL;

  /* if (newpos!=NULL) free(newpos); */
  m= M->rowsize;
  d= M->colsize;
  *newpos=(long *)calloc(m+1,sizeof(long));  
  newpos1r=(long *)calloc(m+1,sizeof(long));  
  if (m>=0 && d>=0){
    M1=ddf_MatrixNormalizedSortedCopy(M,&newpos1);
    for (i=1; i<=m;i++) newpos1r[newpos1[i]]=i;  /* reverse of newpos1 */
    M2=ddf_MatrixUniqueCopy(M1,&newpos2);
    set_emptyset(M2->linset);
    for(i=1; i<=m; i++) {
      if (newpos2[newpos1[i]]>0){
         printf("newpos1[%ld]=%ld, newpos2[newpos1[%ld]]=%ld\n",i,newpos1[i], i,newpos2[newpos1[i]]);
         if (set_member(i,M->linset)) set_addelem(M2->linset, newpos2[newpos1[i]]);
         (*newpos)[i]=newpos2[newpos1[i]];
      } else {
         (*newpos)[i]=-newpos1r[-newpos2[newpos1[i]]];
      }
    }
  ddf_FreeMatrix(M1);free(newpos1);free(newpos2);free(newpos1r);
  }
  
  return M2;
}

ddf_MatrixPtr ddf_MatrixSortedUniqueCopy(ddf_MatrixPtr M,ddf_rowindex *newpos)  /* 094 */
{ 
  /* Same as ddf_MatrixNormalizedSortedUniqueCopy except that it returns a unnormalized origial data
     with original ordering.
  */
  ddf_MatrixPtr M1=NULL,M2=NULL;
  ddf_rowrange m,i,k,ii;
  ddf_colrange d;
  ddf_rowindex newpos1=NULL,newpos1r=NULL,newpos2=NULL;

  /* if (newpos!=NULL) free(newpos); */
  m= M->rowsize;
  d= M->colsize;
  *newpos=(long *)calloc(m+1,sizeof(long));  
  newpos1r=(long *)calloc(m+1,sizeof(long));  
  if (m>=0 && d>=0){
    M1=ddf_MatrixNormalizedSortedCopy(M,&newpos1);
    for (i=1; i<=m;i++) newpos1r[newpos1[i]]=i;  /* reverse of newpos1 */
    M2=ddf_MatrixUniqueCopy(M1,&newpos2);
    ddf_FreeMatrix(M1);
    set_emptyset(M2->linset);
    for(i=1; i<=m; i++) {
      if (newpos2[newpos1[i]]>0){
         if (set_member(i,M->linset)) set_addelem(M2->linset, newpos2[newpos1[i]]);
         (*newpos)[i]=newpos2[newpos1[i]];
      } else {
         (*newpos)[i]=-newpos1r[-newpos2[newpos1[i]]];
      }
    }

    ii=0;
    set_emptyset(M2->linset);
    for (i = 1; i<=m ; i++) {
      k=(*newpos)[i];
      if (k>0) {
        ii+=1;
        (*newpos)[i]=ii;
        ddf_CopyArow(M2->matrix[ii-1],M->matrix[i-1],d);
        if (set_member(i,M->linset)) set_addelem(M2->linset, ii);
      }
    }

    free(newpos1);free(newpos2);free(newpos1r);
  }
  
  return M2;
}

ddf_MatrixPtr ddf_AppendMatrix(ddf_MatrixPtr M1, ddf_MatrixPtr M2)
{
  return ddf_MatrixAppend(M1,M2);
}

int ddf_MatrixAppendTo(ddf_MatrixPtr *M1, ddf_MatrixPtr M2)
{
  ddf_MatrixPtr M=NULL;
  ddf_rowrange i, m,m1,m2;
  ddf_colrange j, d,d1,d2;
  ddf_boolean success=0;
  
  m1=(*M1)->rowsize;
  d1=(*M1)->colsize;
  m2=M2->rowsize;
  d2=M2->colsize;

  m=m1+m2;
  d=d1;

  if (d1>=0 && d1==d2 && m1>=0 && m2>=0){
    M=ddf_CreateMatrix(m, d);
    ddf_CopyAmatrix(M->matrix, (*M1)->matrix, m1, d);
    ddf_CopyArow(M->rowvec, (*M1)->rowvec, d);
    for (i=0; i<m1; i++){
      if (set_member(i+1,(*M1)->linset)) set_addelem(M->linset,i+1);
    }
    for (i=0; i<m2; i++){
       for (j=0; j<d; j++)
         ddf_set(M->matrix[m1+i][j],M2->matrix[i][j]);
         /* append the second matrix */
       if (set_member(i+1,M2->linset)) set_addelem(M->linset,m1+i+1);
    }
    M->numbtype=(*M1)->numbtype;
    M->representation=(*M1)->representation;
    M->objective=(*M1)->objective;
    ddf_FreeMatrix(*M1);
    *M1=M;
    success=1;
  }
  return success;
}

int ddf_MatrixRowRemove(ddf_MatrixPtr *M, ddf_rowrange r) /* 092 */
{
  ddf_rowrange i,m;
  ddf_colrange d;
  ddf_boolean success=0;
  
  m=(*M)->rowsize;
  d=(*M)->colsize;

  if (r >= 1 && r <=m){
    (*M)->rowsize=m-1;
    ddf_FreeArow(d, (*M)->matrix[r-1]);
    set_delelem((*M)->linset,r);
    /* slide the row headers */
    for (i=r; i<m; i++){
      (*M)->matrix[i-1]=(*M)->matrix[i];
      if (set_member(i+1, (*M)->linset)){
        set_delelem((*M)->linset,i+1);
        set_addelem((*M)->linset,i);
      }
    }
    success=1;
  }
  return success;
}

int ddf_MatrixRowRemove2(ddf_MatrixPtr *M, ddf_rowrange r, ddf_rowindex *newpos) /* 094 */
{
  ddf_rowrange i,m;
  ddf_colrange d;
  ddf_boolean success=0;
  ddf_rowindex roworder;
  
  m=(*M)->rowsize;
  d=(*M)->colsize;

  if (r >= 1 && r <=m){
    roworder=(long *)calloc(m+1,sizeof(long));
    (*M)->rowsize=m-1;
    ddf_FreeArow(d, (*M)->matrix[r-1]);
    set_delelem((*M)->linset,r);
    /* slide the row headers */
    for (i=1; i<r; i++) roworder[i]=i;
    roworder[r]=0; /* meaning it is removed */
    for (i=r; i<m; i++){
      (*M)->matrix[i-1]=(*M)->matrix[i];
      roworder[i+1]=i;
      if (set_member(i+1, (*M)->linset)){
        set_delelem((*M)->linset,i+1);
        set_addelem((*M)->linset,i);
      }
    }
    success=1;
  }
  return success;
}

ddf_MatrixPtr ddf_MatrixSubmatrix(ddf_MatrixPtr M, ddf_rowset delset) /* 092 */
{
  ddf_MatrixPtr Msub=NULL;
  ddf_rowrange i,isub=1, m,msub;
  ddf_colrange d;
 
  m= M->rowsize;
  d= M->colsize;
  msub=m;
  if (m >=0 && d >=0){
    for (i=1; i<=m; i++) {
       if (set_member(i,delset)) msub-=1;
    }
    Msub=ddf_CreateMatrix(msub, d);
    for (i=1; i<=m; i++){
      if (!set_member(i,delset)){
        ddf_CopyArow(Msub->matrix[isub-1], M->matrix[i-1], d);
        if (set_member(i, M->linset)){
          set_addelem(Msub->linset,isub);
        }
        isub++;
      }
    }
    ddf_CopyArow(Msub->rowvec, M->rowvec, d);
    Msub->numbtype=M->numbtype;
    Msub->representation=M->representation;
    Msub->objective=M->objective;
  }
  return Msub;
}

ddf_MatrixPtr ddf_MatrixSubmatrix2(ddf_MatrixPtr M, ddf_rowset delset,ddf_rowindex *newpos) /* 092 */
{ /* returns a pointer to a new matrix which is a submatrix of M with rows in delset
  removed.  *newpos[i] returns the position of the original row i in the new matrix.
  It is -1 if and only if it is deleted.
  */
  
  ddf_MatrixPtr Msub=NULL;
  ddf_rowrange i,isub=1, m,msub;
  ddf_colrange d;
  ddf_rowindex roworder;

  m= M->rowsize;
  d= M->colsize;
  msub=m;
  if (m >=0 && d >=0){
    roworder=(long *)calloc(m+1,sizeof(long));
    for (i=1; i<=m; i++) {
       if (set_member(i,delset)) msub-=1;
    }
    Msub=ddf_CreateMatrix(msub, d);
    for (i=1; i<=m; i++){
      if (set_member(i,delset)){
        roworder[i]=0; /* zero means the row i is removed */
      } else {
        ddf_CopyArow(Msub->matrix[isub-1], M->matrix[i-1], d);
        if (set_member(i, M->linset)){
          set_addelem(Msub->linset,isub);
        }
        roworder[i]=isub;
        isub++;
      }
    }
    *newpos=roworder;
    ddf_CopyArow(Msub->rowvec, M->rowvec, d);
    Msub->numbtype=M->numbtype;
    Msub->representation=M->representation;
    Msub->objective=M->objective;
  }
  return Msub;
}


ddf_MatrixPtr ddf_MatrixSubmatrix2L(ddf_MatrixPtr M, ddf_rowset delset,ddf_rowindex *newpos) /* 094 */
{ /* This is same as ddf_MatrixSubmatrix2 except that the linearity rows will be shifted up
     so that they are at the top of the matrix.
  */
  ddf_MatrixPtr Msub=NULL;
  ddf_rowrange i,iL, iI, m,msub;
  ddf_colrange d;
  ddf_rowindex roworder;

  m= M->rowsize;
  d= M->colsize;
  msub=m;
  if (m >=0 && d >=0){
    roworder=(long *)calloc(m+1,sizeof(long));
    for (i=1; i<=m; i++) {
       if (set_member(i,delset)) msub-=1;
    }
    Msub=ddf_CreateMatrix(msub, d);
    iL=1; iI=set_card(M->linset)+1;  /* starting positions */
    for (i=1; i<=m; i++){
      if (set_member(i,delset)){
        roworder[i]=0; /* zero means the row i is removed */
      } else {
        if (set_member(i,M->linset)){
          ddf_CopyArow(Msub->matrix[iL-1], M->matrix[i-1], d);
          set_delelem(Msub->linset,i);
          set_addelem(Msub->linset,iL);
          roworder[i]=iL;
          iL+=1;
        } else {
          ddf_CopyArow(Msub->matrix[iI-1], M->matrix[i-1], d);
          roworder[i]=iI;
          iI+=1;
        }
       }
    }
    *newpos=roworder;
    ddf_CopyArow(Msub->rowvec, M->rowvec, d);
    Msub->numbtype=M->numbtype;
    Msub->representation=M->representation;
    Msub->objective=M->objective;
  }
  return Msub;
}

int ddf_MatrixRowsRemove(ddf_MatrixPtr *M, ddf_rowset delset) /* 094 */
{
  ddf_MatrixPtr Msub=NULL;
  int success;
  
  Msub=ddf_MatrixSubmatrix(*M, delset);
  ddf_FreeMatrix(*M);
  *M=Msub;
  success=1;
  return success;
}

int ddf_MatrixRowsRemove2(ddf_MatrixPtr *M, ddf_rowset delset,ddf_rowindex *newpos) /* 094 */
{
  ddf_MatrixPtr Msub=NULL;
  int success;
  
  Msub=ddf_MatrixSubmatrix2(*M, delset,newpos);
  ddf_FreeMatrix(*M);
  *M=Msub;
  success=1;
  return success;
}

int ddf_MatrixShiftupLinearity(ddf_MatrixPtr *M,ddf_rowindex *newpos) /* 094 */
{
  ddf_MatrixPtr Msub=NULL;
  int success;
  ddf_rowset delset;
  
  set_initialize(&delset,(*M)->rowsize);  /* emptyset */
  Msub=ddf_MatrixSubmatrix2L(*M, delset,newpos);
  ddf_FreeMatrix(*M);
  *M=Msub;
  
  set_free(delset);
  success=1;
  return success;
}

ddf_PolyhedraPtr ddf_CreatePolyhedraData(ddf_rowrange m, ddf_colrange d)
{
  ddf_rowrange i;
  ddf_PolyhedraPtr poly=NULL;

  poly=(ddf_PolyhedraPtr) malloc (sizeof(ddf_PolyhedraType));
  poly->child       =NULL; /* this links the homogenized cone data */
  poly->m           =m;
  poly->d           =d;  
  poly->n           =-1;  /* the size of output is not known */
  poly->m_alloc     =m+2; /* the allocated row size of matrix A */
  poly->d_alloc     =d;   /* the allocated col size of matrix A */
  poly->ldim		=0;   /* initialize the linearity dimension */
  poly->numbtype=ddf_Real;
  ddf_InitializeAmatrix(poly->m_alloc,poly->d_alloc,&(poly->A));
  ddf_InitializeArow(d,&(poly->c));           /* cost vector */
  poly->representation       =ddf_Inequality;
  poly->homogeneous =ddf_FALSE;

  poly->EqualityIndex=(int *)calloc(m+2, sizeof(int));  
    /* size increased to m+2 in 092b because it is used by the child cone, 
       This is a bug fix suggested by Thao Dang. */
    /* ith component is 1 if it is equality, -1 if it is strict inequality, 0 otherwise. */
  for (i = 0; i <= m+1; i++) poly->EqualityIndex[i]=0;

  poly->IsEmpty                 = -1;  /* initially set to -1, neither TRUE nor FALSE, meaning unknown */
  
  poly->NondegAssumed           = ddf_FALSE;
  poly->InitBasisAtBottom       = ddf_FALSE;
  poly->RestrictedEnumeration   = ddf_FALSE;
  poly->RelaxedEnumeration      = ddf_FALSE;

  poly->AincGenerated=ddf_FALSE;  /* Ainc is a set array to store the input incidence. */

  return poly;
}

ddf_boolean ddf_InitializeConeData(ddf_rowrange m, ddf_colrange d, ddf_ConePtr *cone)
{
  ddf_boolean success=ddf_TRUE;
  ddf_colrange j;

  (*cone)=(ddf_ConePtr)calloc(1, sizeof(ddf_ConeType));

/* INPUT: A given representation of a cone: inequality */
  (*cone)->m=m;
  (*cone)->d=d;
  (*cone)->m_alloc=m+2; /* allocated row size of matrix A */
  (*cone)->d_alloc=d;   /* allocated col size of matrix A, B and Bsave */
  (*cone)->numbtype=ddf_Real;
  (*cone)->parent=NULL;

/* CONTROL: variables to control computation */
  (*cone)->Iteration=0;

  (*cone)->HalfspaceOrder=ddf_LexMin;

  (*cone)->ArtificialRay=NULL;
  (*cone)->FirstRay=NULL;
  (*cone)->LastRay=NULL; /* The second description: Generator */
  (*cone)->PosHead=NULL;
  (*cone)->ZeroHead=NULL;
  (*cone)->NegHead=NULL;
  (*cone)->PosLast=NULL;
  (*cone)->ZeroLast=NULL;
  (*cone)->NegLast=NULL;
  (*cone)->RecomputeRowOrder  = ddf_TRUE;
  (*cone)->PreOrderedRun      = ddf_FALSE;
  set_initialize(&((*cone)->GroundSet),(*cone)->m_alloc);
  set_initialize(&((*cone)->EqualitySet),(*cone)->m_alloc);
  set_initialize(&((*cone)->NonequalitySet),(*cone)->m_alloc);
  set_initialize(&((*cone)->AddedHalfspaces),(*cone)->m_alloc);
  set_initialize(&((*cone)->WeaklyAddedHalfspaces),(*cone)->m_alloc);
  set_initialize(&((*cone)->InitialHalfspaces),(*cone)->m_alloc);
  (*cone)->RayCount=0;
  (*cone)->FeasibleRayCount=0;
  (*cone)->WeaklyFeasibleRayCount=0;
  (*cone)->TotalRayCount=0;
  (*cone)->ZeroRayCount=0;
  (*cone)->EdgeCount=0;
  (*cone)->TotalEdgeCount=0;
  (*cone)->count_int=0;
  (*cone)->count_int_good=0;
  (*cone)->count_int_bad=0;
  (*cone)->rseed=1;  /* random seed for random row permutation */
 
  ddf_InitializeBmatrix((*cone)->d_alloc, &((*cone)->B));
  ddf_InitializeBmatrix((*cone)->d_alloc, &((*cone)->Bsave));
  ddf_InitializeAmatrix((*cone)->m_alloc,(*cone)->d_alloc,&((*cone)->A));

  (*cone)->Edges
     =(ddf_AdjacencyType**) calloc((*cone)->m_alloc,sizeof(ddf_AdjacencyType*));
  for (j=0; j<(*cone)->m_alloc; j++) (*cone)->Edges[j]=NULL; /* 094h */
  (*cone)->InitialRayIndex=(long*)calloc(d+1,sizeof(long));
  (*cone)->OrderVector=(long*)calloc((*cone)->m_alloc+1,sizeof(long));

  (*cone)->newcol=(long*)calloc(((*cone)->d)+1,sizeof(long));
  for (j=0; j<=(*cone)->d; j++) (*cone)->newcol[j]=j;  /* identity map, initially */
  (*cone)->LinearityDim = -2; /* -2 if it is not computed */
  (*cone)->ColReduced   = ddf_FALSE;
  (*cone)->d_orig = d;

/* STATES: variables to represent current state. */
/*(*cone)->Error;
  (*cone)->CompStatus;
  (*cone)->starttime;
  (*cone)->endtime;
*/
    
  return success;
}

ddf_ConePtr ddf_ConeDataLoad(ddf_PolyhedraPtr poly)
{
  ddf_ConePtr cone=NULL;
  ddf_colrange d,j;
  ddf_rowrange m,i;

  m=poly->m;
  d=poly->d;
  if (!(poly->homogeneous) && poly->representation==ddf_Inequality){
    m=poly->m+1;
  }
  poly->m1=m;

  ddf_InitializeConeData(m, d, &cone);
  cone->representation=poly->representation;

/* Points to the original polyhedra data, and reversely */
  cone->parent=poly;
  poly->child=cone;

  for (i=1; i<=poly->m; i++)
    for (j=1; j<=cone->d; j++)
      ddf_set(cone->A[i-1][j-1],poly->A[i-1][j-1]);  
  
  if (poly->representation==ddf_Inequality && !(poly->homogeneous)){
    ddf_set(cone->A[m-1][0],ddf_one);
    for (j=2; j<=d; j++) ddf_set(cone->A[m-1][j-1],ddf_purezero);
  }

  return cone;
}

void ddf_SetLinearity(ddf_MatrixPtr M, char *line)
{
  int i=0;
  ddf_rowrange eqsize,var;
  char *next;
  const char ct[]=", ";  /* allows separators "," and " ". */

  next=strtok(line,ct);
  eqsize=atol(next); 
  while (i < eqsize && (next=strtok(NULL,ct))!=NULL) {
     var=atol(next);
     set_addelem(M->linset,var); i++;
  }
  if (i!=eqsize) {
    fprintf(stderr,"* Warning: there are inconsistencies in linearity setting.\n");
  }
  return;
}

ddf_MatrixPtr ddf_PolyFile2Matrix (FILE *f, ddf_ErrorType *Error)
{
  ddf_MatrixPtr M=NULL;
  ddf_rowrange m_input,i;
  ddf_colrange d_input,j;
  ddf_RepresentationType rep=ddf_Inequality;
  myfloat value;
  ddf_boolean found=ddf_FALSE, newformat=ddf_FALSE, successful=ddf_FALSE, linearity=ddf_FALSE;
  char command[ddf_linelenmax], comsave[ddf_linelenmax], numbtype[ddf_wordlenmax];
  ddf_NumberType NT;
#if !defined(ddf_GMPRATIONAL)
  double rvalue;
#endif

  ddf_init(value);
  (*Error)=ddf_NoError;
  while (!found)
  {
    if (fscanf(f,"%s",command)==EOF) {
      (*Error)=ddf_ImproperInputFormat;
      goto _L99;
    }
    else {
      if (strncmp(command, "V-representation", 16)==0) {
        rep=ddf_Generator; newformat=ddf_TRUE;
      }
      if (strncmp(command, "H-representation", 16)==0){
        rep=ddf_Inequality; newformat=ddf_TRUE;
      }
      if (strncmp(command, "partial_enum", 12)==0 || 
          strncmp(command, "equality", 8)==0  ||
          strncmp(command, "linearity", 9)==0 ) {
        linearity=ddf_TRUE;
        fgets(comsave,ddf_linelenmax,f);
      }
      if (strncmp(command, "begin", 5)==0) found=ddf_TRUE;
    }
  }
  fscanf(f, "%ld %ld %s", &m_input, &d_input, numbtype);
  fprintf(stderr,"size = %ld x %ld\nNumber Type = %s\n", m_input, d_input, numbtype);
  NT=ddf_GetNumberType(numbtype);
  if (NT==ddf_Unknown) {
      (*Error)=ddf_ImproperInputFormat;
      goto _L99;
    } 
  M=ddf_CreateMatrix(m_input, d_input);
  M->representation=rep;
  M->numbtype=NT;

  for (i = 1; i <= m_input; i++) {
    for (j = 1; j <= d_input; j++) {
      if (NT==ddf_Real) {
#if defined ddf_GMPRATIONAL
        *Error=ddf_NoRealNumberSupport;
        goto _L99;
#else
        fscanf(f, "%lf", &rvalue);
        ddf_set_d(value, rvalue);
#endif
      } else {
        ddf_fread_rational_value (f, value);
      }
      ddf_set(M->matrix[i-1][j - 1],value);
      if (ddf_debug) {fprintf(stderr,"a(%3ld,%5ld) = ",i,j); ddf_WriteNumber(stderr,value);}
    }  /*of j*/
  }  /*of i*/
  if (fscanf(f,"%s",command)==EOF) {
   	 (*Error)=ddf_ImproperInputFormat;
  	 goto _L99;
  }
  else if (strncmp(command, "end", 3)!=0) {
     if (ddf_debug) fprintf(stderr,"'end' missing or illegal extra data: %s\n",command);
     (*Error)=ddf_ImproperInputFormat;
     goto _L99;
  }
  
  successful=ddf_TRUE;
  if (linearity) {
    ddf_SetLinearity(M,comsave);
  }
  while (!feof(f)) {
    fscanf(f,"%s", command);
    ddf_ProcessCommandLine(f, M, command);
    fgets(command,ddf_linelenmax,f); /* skip the CR/LF */
  } 

_L99: ;
  ddf_clear(value);
  /* if (f!=NULL) fclose(f); */
  return M;
}


ddf_PolyhedraPtr ddf_DDMatrix2Poly(ddf_MatrixPtr M, ddf_ErrorType *err)
{
  ddf_rowrange i;
  ddf_colrange j;
  ddf_PolyhedraPtr poly=NULL;

  *err=ddf_NoError;
  if (M->rowsize<0 || M->colsize<0){
    *err=ddf_NegativeMatrixSize;
    goto _L99;
  }
  poly=ddf_CreatePolyhedraData(M->rowsize, M->colsize);
  poly->representation=M->representation;
  poly->homogeneous=ddf_TRUE;

  for (i = 1; i <= M->rowsize; i++) {
    if (set_member(i, M->linset)) {
      poly->EqualityIndex[i]=1;
    }
    for (j = 1; j <= M->colsize; j++) {
      ddf_set(poly->A[i-1][j-1], M->matrix[i-1][j-1]);
      if (j==1 && ddf_Nonzero(M->matrix[i-1][j-1])) poly->homogeneous = ddf_FALSE;
    }  /*of j*/
  }  /*of i*/
  ddf_DoubleDescription(poly,err);
_L99:
  return poly; 
}

ddf_PolyhedraPtr ddf_DDMatrix2Poly2(ddf_MatrixPtr M, ddf_RowOrderType horder, ddf_ErrorType *err)
{
  ddf_rowrange i;
  ddf_colrange j;
  ddf_PolyhedraPtr poly=NULL;

  *err=ddf_NoError;
  if (M->rowsize<0 || M->colsize<0){
    *err=ddf_NegativeMatrixSize;
    goto _L99;
  }
  poly=ddf_CreatePolyhedraData(M->rowsize, M->colsize);
  poly->representation=M->representation;
  poly->homogeneous=ddf_TRUE;

  for (i = 1; i <= M->rowsize; i++) {
    if (set_member(i, M->linset)) {
      poly->EqualityIndex[i]=1;
    }
    for (j = 1; j <= M->colsize; j++) {
      ddf_set(poly->A[i-1][j-1], M->matrix[i-1][j-1]);
      if (j==1 && ddf_Nonzero(M->matrix[i-1][j-1])) poly->homogeneous = ddf_FALSE;
    }  /*of j*/
  }  /*of i*/
  ddf_DoubleDescription2(poly, horder, err);
_L99:
  return poly; 
}

void ddf_MatrixIntegerFilter(ddf_MatrixPtr M)
{   /* setting an almost integer to the integer. */
  ddf_rowrange i;
  ddf_colrange j;
  myfloat x;

  ddf_init(x);
  for (i=0; i< M->rowsize; i++)
    for (j=0; j< M->colsize; j++){
       ddf_SnapToInteger(x, M->matrix[i][j]);
       ddf_set(M->matrix[i][j],x);
    }
  ddf_clear(x);
}

void ddf_CopyRay(myfloat *a, ddf_colrange d_origsize, ddf_RayPtr RR, 
  ddf_RepresentationType rep, ddf_colindex reducedcol)
{
  long j,j1;
  myfloat b;

  ddf_init(b);
  for (j = 1; j <= d_origsize; j++){
    j1=reducedcol[j];
    if (j1>0){
      ddf_set(a[j-1],RR->Ray[j1-1]); 
        /* the original column j is mapped to j1, and thus
           copy the corresponding component */
    } else {
      ddf_set(a[j-1],ddf_purezero);  
        /* original column is redundant and removed for computation */
    }
  }

  ddf_set(b,a[0]);
  if (rep==ddf_Generator && ddf_Nonzero(b)){
    ddf_set(a[0],ddf_one);
    for (j = 2; j <= d_origsize; j++)
       ddf_div(a[j-1],a[j-1],b);    /* normalization for generators */
  }
  ddf_clear(b);
}

void ddf_WriteRay(FILE *f, ddf_colrange d_origsize, ddf_RayPtr RR, ddf_RepresentationType rep, ddf_colindex reducedcol)
{
  ddf_colrange j;
  static ddf_colrange d_last=0;
  static ddf_Arow a;

  if (d_last< d_origsize){
    if (d_last>0) free(a);
    ddf_InitializeArow(d_origsize+1, &a);
    d_last=d_origsize+1;
  }

 ddf_CopyRay(a, d_origsize, RR, rep, reducedcol);
  for (j = 0; j < d_origsize; j++) ddf_WriteNumber(f, a[j]);
  fprintf(f, "\n");
}

void ddf_WriteArow(FILE *f, ddf_Arow a, ddf_colrange d)
{
  ddf_colrange j;

  for (j = 0; j < d; j++) ddf_WriteNumber(f, a[j]);
  fprintf(f, "\n");
}

void ddf_WriteAmatrix(FILE *f, ddf_Amatrix A, long rowmax, long colmax)
{
  long i,j;

  if (A==NULL){
    fprintf(f, "WriteAmatrix: The requested matrix is empty\n");
    goto _L99;
  }
  fprintf(f, "begin\n");
#if defined ddf_GMPRATIONAL
  fprintf(f, " %ld %ld rational\n",rowmax, colmax);
#else
  fprintf(f, " %ld %ld real\n",rowmax, colmax);
#endif
  for (i=1; i <= rowmax; i++) {
    for (j=1; j <= colmax; j++) {
      ddf_WriteNumber(f, A[i-1][j-1]);
    }
    fprintf(f,"\n");
  }
  fprintf(f, "end\n");
_L99:;
}

void ddf_WriteBmatrix(FILE *f, ddf_colrange d_size, ddf_Bmatrix B)
{
  ddf_colrange j1, j2;

  if (B==NULL){
    fprintf(f, "WriteBmatrix: The requested matrix is empty\n");
    goto _L99;
  }
  for (j1 = 0; j1 < d_size; j1++) {
    for (j2 = 0; j2 < d_size; j2++) {
      ddf_WriteNumber(f, B[j1][j2]);
    }  /*of j2*/
    putc('\n', f);
  }  /*of j1*/
  putc('\n', f);
_L99:;
}

void ddf_WriteSetFamily(FILE *f, ddf_SetFamilyPtr F)
{
  ddf_bigrange i;

  if (F==NULL){
    fprintf(f, "WriteSetFamily: The requested family is empty\n");
    goto _L99;
  }
  fprintf(f,"begin\n");
  fprintf(f,"  %ld    %ld\n", F->famsize, F->setsize);
  for (i=0; i<F->famsize; i++) {
    fprintf(f, " %ld %ld : ", i+1, set_card(F->set[i]));
    set_fwrite(f, F->set[i]);
  }
  fprintf(f,"end\n");
_L99:;
}

void ddf_WriteSetFamilyCompressed(FILE *f, ddf_SetFamilyPtr F)
{
  ddf_bigrange i,card;

  if (F==NULL){
    fprintf(f, "WriteSetFamily: The requested family is empty\n");
    goto _L99;
  }
  fprintf(f,"begin\n");
  fprintf(f,"  %ld    %ld\n", F->famsize, F->setsize);
  for (i=0; i<F->famsize; i++) {
    card=set_card(F->set[i]);
    if (F->setsize - card >= card){
      fprintf(f, " %ld %ld : ", i+1, card);
      set_fwrite(f, F->set[i]);
    } else {
      fprintf(f, " %ld %ld : ", i+1, -card);
      set_fwrite_compl(f, F->set[i]);
    }
  }
  fprintf(f,"end\n");
_L99:;
}

void ddf_WriteMatrix(FILE *f, ddf_MatrixPtr M)
{
  ddf_rowrange i, linsize;

  if (M==NULL){
    fprintf(f, "WriteMatrix: The requested matrix is empty\n");
    goto _L99;
  }
  switch (M->representation) {
    case ddf_Inequality:
      fprintf(f, "H-representation\n"); break;
    case ddf_Generator:
      fprintf(f, "V-representation\n"); break;
    case ddf_Unspecified:
      break;
  }
  linsize=set_card(M->linset);
  if (linsize>0) {
    fprintf(f, "linearity %ld ", linsize);
    for (i=1; i<=M->rowsize; i++) 
      if (set_member(i, M->linset)) fprintf(f, " %ld", i);
    fprintf(f, "\n");
  }
  ddf_WriteAmatrix(f, M->matrix, M->rowsize, M->colsize);
  if (M->objective!=ddf_LPnone){
    if (M->objective==ddf_LPmax)
      fprintf(f, "maximize\n");
    else
      fprintf(f, "minimize\n");      
    ddf_WriteArow(f, M->rowvec, M->colsize);
  }
_L99:;
}

void ddf_WriteLP(FILE *f, ddf_LPPtr lp)
{
  if (lp==NULL){
    fprintf(f, "WriteLP: The requested lp is empty\n");
    goto _L99;
  }
  fprintf(f, "H-representation\n");

  ddf_WriteAmatrix(f, lp->A, (lp->m)-1, lp->d);
  if (lp->objective!=ddf_LPnone){
    if (lp->objective==ddf_LPmax)
      fprintf(f, "maximize\n");
    else
      fprintf(f, "minimize\n");      
    ddf_WriteArow(f, lp->A[lp->objrow-1], lp->d);
  }
_L99:;
}


void ddf_SnapToInteger(myfloat y, myfloat x)
{
 /* this is broken.  It does nothing. */
   ddf_set(y,x);
}


void ddf_WriteReal(FILE *f, myfloat x)
{
  long ix1,ix2,ix;
  double ax;

  ax=ddf_get_d(x);  
  ix1= (long) (fabs(ax) * 10000. + 0.5);
  ix2= (long) (fabs(ax) + 0.5);
  ix2= ix2*10000;
  if ( ix1 == ix2) {
    if (ddf_Positive(x)) {
      ix = (long)(ax + 0.5);
    } else {
      ix = (long)(-ax + 0.5);
      ix = -ix;
    }
    fprintf(f, " %2ld", ix);
  } else
    fprintf(f, " % .9E",ax);
}

void ddf_WriteNumber(FILE *f, myfloat x)
{
#if defined ddf_GMPRATIONAL
  fprintf(f," ");
  mpq_out_str(f, 10, x);
#else
  ddf_WriteReal(f, x);
#endif
}


void ddf_WriteIncidence(FILE *f, ddf_PolyhedraPtr poly)
{
  ddf_SetFamilyPtr I;

  switch (poly->representation) {
    case ddf_Inequality:
       fprintf(f, "ecd_file: Incidence of generators and inequalities\n");
      break;
    case ddf_Generator:
       fprintf(f, "icd_file: Incidence of inequalities and generators\n");
      break;

    default:
      break;
  }
  I=ddf_CopyIncidence(poly);
  ddf_WriteSetFamilyCompressed(f,I);
  ddf_FreeSetFamily(I);
}

void ddf_WriteAdjacency(FILE *f, ddf_PolyhedraPtr poly)
{
  ddf_SetFamilyPtr A;

  switch (poly->representation) {
    case ddf_Inequality:
       fprintf(f, "ead_file: Adjacency of generators\n");
      break;
    case ddf_Generator:
       fprintf(f, "iad_file: Adjacency of inequalities\n");
      break;

    default:
      break;
  }
  A=ddf_CopyAdjacency(poly);
  ddf_WriteSetFamilyCompressed(f,A);
  ddf_FreeSetFamily(A);
}


void ddf_ComputeAinc(ddf_PolyhedraPtr poly)
{
/* This generates the input incidence array poly->Ainc, and
   two sets: poly->Ared, poly->Adom. 
*/
  ddf_bigrange k;
  ddf_rowrange i,m1;
  ddf_colrange j;
  ddf_boolean redundant;
  ddf_MatrixPtr M=NULL;
  myfloat sum,temp;

  ddf_init(sum); ddf_init(temp);
  if (poly->AincGenerated==ddf_TRUE) goto _L99;

  M=ddf_CopyOutput(poly);
  poly->n=M->rowsize;
  m1=poly->m1;  
   /* this number is same as poly->m, except when
      poly is given by nonhomogeneous inequalty:
      !(poly->homogeneous) && poly->representation==Inequality,
      it is poly->m+1.   See ddf_ConeDataLoad.
   */
  poly->Ainc=(set_type*)calloc(m1, sizeof(set_type));
  for(i=1; i<=m1; i++) set_initialize(&(poly->Ainc[i-1]),poly->n);
  set_initialize(&(poly->Ared), m1); 
  set_initialize(&(poly->Adom), m1); 

  for (k=1; k<=poly->n; k++){
    for (i=1; i<=poly->m; i++){
      ddf_set(sum,ddf_purezero);
      for (j=1; j<=poly->d; j++){
        ddf_mul(temp,poly->A[i-1][j-1],M->matrix[k-1][j-1]);
        ddf_add(sum,sum,temp);
      }
      if (ddf_EqualToZero(sum)) {
        set_addelem(poly->Ainc[i-1], k);
      }
    }
    if (!(poly->homogeneous) && poly->representation==ddf_Inequality){
      if (ddf_EqualToZero(M->matrix[k-1][0])) {
        set_addelem(poly->Ainc[m1-1], k);  /* added infinity inequality (1,0,0,...,0) */
      }
    }
  }

  for (i=1; i<=m1; i++){
    if (set_card(poly->Ainc[i-1])==M->rowsize){
      set_addelem(poly->Adom, i);
    }  
  }
  for (i=m1; i>=1; i--){
    if (set_card(poly->Ainc[i-1])==0){
      redundant=ddf_TRUE;
      set_addelem(poly->Ared, i);
    }else {
      redundant=ddf_FALSE;
      for (k=1; k<=m1; k++) {
        if (k!=i && !set_member(k, poly->Ared)  && !set_member(k, poly->Adom) && 
            set_subset(poly->Ainc[i-1], poly->Ainc[k-1])){
          if (!redundant){
            redundant=ddf_TRUE;
          }
          set_addelem(poly->Ared, i);
        }
      }
    }
  }
  ddf_FreeMatrix(M);
  poly->AincGenerated=ddf_TRUE;
_L99:;
  ddf_clear(sum);  ddf_clear(temp);
}

ddf_boolean ddf_InputAdjacentQ(ddf_PolyhedraPtr poly, 
  ddf_rowrange i1, ddf_rowrange i2)
/* Before calling this function, RedundantSet must be 
   a set of row indices whose removal results in a minimal
   nonredundant system to represent the input polyhedron,
   DominantSet must be the set of row indices which are
   active at every extreme points/rays.
*/
{
  ddf_boolean adj=ddf_TRUE;
  ddf_rowrange i;
  static set_type common;
  static long lastn=0;

  if (poly->AincGenerated==ddf_FALSE) ddf_ComputeAinc(poly);
  if (lastn!=poly->n){
    if (lastn >0) set_free(common);
    set_initialize(&common, poly->n);
    lastn=poly->n;
  }
  if (set_member(i1, poly->Ared) || set_member(i2, poly->Ared)){
    adj=ddf_FALSE;
    goto _L99;
  }
  if (set_member(i1, poly->Adom) || set_member(i2, poly->Adom)){
  // dominant inequality is considered adjacencent to all others.
    adj=ddf_TRUE;
    goto _L99;
  }
  set_int(common, poly->Ainc[i1-1], poly->Ainc[i2-1]);
  i=0;
  while (i<poly->m1 && adj==ddf_TRUE){ 
    i++; 
    if (i!=i1 && i!=i2 && !set_member(i, poly->Ared) &&
        !set_member(i, poly->Adom) && set_subset(common,poly->Ainc[i-1])){
      adj=ddf_FALSE;
    }
  }
_L99:;
  return adj;
} 


void ddf_WriteInputIncidence(FILE *f, ddf_PolyhedraPtr poly)
{
  ddf_SetFamilyPtr I;

  if (poly->AincGenerated==ddf_FALSE) ddf_ComputeAinc(poly);
  switch (poly->representation) {
  case ddf_Inequality:
    fprintf(f,"icd_file: Incidence of inequalities and generators\n");
    break;

  case ddf_Generator:
   fprintf(f,"ecd_file: Incidence of generators and inequalities\n");
    break;

  default:
    break;
  }

  I=ddf_CopyInputIncidence(poly);
  ddf_WriteSetFamilyCompressed(f,I);
  ddf_FreeSetFamily(I);
}

void ddf_WriteInputAdjacency(FILE *f, ddf_PolyhedraPtr poly)
{
  ddf_SetFamilyPtr A;

  if (poly->AincGenerated==ddf_FALSE){
    ddf_ComputeAinc(poly);
  }
  switch (poly->representation) {
  case ddf_Inequality:
    fprintf(f, "iad_file: Adjacency of inequalities\n");
    break;

  case ddf_Generator:
    fprintf(f, "ead_file: Adjacency of generators\n");
    break;

  default:
    break;
  }
  A=ddf_CopyInputAdjacency(poly);
  ddf_WriteSetFamilyCompressed(f,A);
  ddf_FreeSetFamily(A);
}


void ddf_WriteProgramDescription(FILE *f)
{
  fprintf(f, "* cddlib: a double description library:%s\n", ddf_DDVERSION);
  fprintf(f, "* compiled for %s arithmetic.\n", ddf_ddf_ARITHMETIC);
  fprintf(f,"* %s\n",ddf_COPYRIGHT);
}

void ddf_WriteTimes(FILE *f, time_t starttime, time_t endtime)
{
  long ptime,ptime_sec,ptime_minu, ptime_hour;
  
/* 
   ptime=difftime(endtime,starttime); 
   This function is ANSI standard, but not available sometime 
*/
  ptime=(endtime - starttime);     
 /* This is to replace the line above, but it may not give 
    correct time in seconds */ 
  ptime_hour=ptime/3600;
  ptime_minu=(ptime-ptime_hour*3600)/60;
  ptime_sec=ptime%60;
  fprintf(f,"* Computation started at %s",asctime(localtime(&starttime)));
  fprintf(f,"*             ended   at %s",asctime(localtime(&endtime)));
  fprintf(f,"* Total processor time = %ld seconds\n",ptime);
  fprintf(f,"*                      = %ld h %ld m %ld s\n",ptime_hour, ptime_minu, ptime_sec);
}

void ddf_WriteDDTimes(FILE *f, ddf_PolyhedraPtr poly)
{
  ddf_WriteTimes(f,poly->child->starttime,poly->child->endtime);
}

void ddf_WriteLPTimes(FILE *f, ddf_LPPtr lp)
{
  ddf_WriteTimes(f,lp->starttime,lp->endtime);
}

void ddf_WriteLPStats(FILE *f)
{
  time_t currenttime;
  
  time(&currenttime);
  
  fprintf(f, "\n*--- Statistics of pivots ---\n");
#if defined ddf_GMPRATIONAL
  fprintf(f, "* f0 = %ld (float basis finding pivots)\n",ddf_statBApivots);
  fprintf(f, "* fc = %ld (float CC pivots)\n",ddf_statCCpivots);
  fprintf(f, "* f1 = %ld (float dual simplex phase I pivots)\n",ddf_statDS1pivots);
  fprintf(f, "* f2 = %ld (float dual simplex phase II pivots)\n",ddf_statDS2pivots);
  fprintf(f, "* f3 = %ld (float anticycling CC pivots)\n",ddf_statACpivots);
  fprintf(f, "* e0 = %ld (exact basis finding pivots)\n",ddf_statBApivots);
  fprintf(f, "* ec = %ld (exact CC pivots)\n",ddf_statCCpivots);
  fprintf(f, "* e1 = %ld (exact dual simplex phase I pivots)\n",ddf_statDS1pivots);
  fprintf(f, "* e2 = %ld (exact dual simplex phase II pivots)\n",ddf_statDS2pivots);
  fprintf(f, "* e3 = %ld (exact anticycling CC pivots)\n",ddf_statACpivots);
  fprintf(f, "* e4 = %ld (exact basis verification pivots)\n",ddf_statBSpivots);
#else
  fprintf(f, "f0 = %ld (float basis finding pivots)\n",ddf_statBApivots);
  fprintf(f, "fc = %ld (float CC pivots)\n",ddf_statCCpivots);
  fprintf(f, "f1 = %ld (float dual simplex phase I pivots)\n",ddf_statDS1pivots);
  fprintf(f, "f2 = %ld (float dual simplex phase II pivots)\n",ddf_statDS2pivots);
  fprintf(f, "f3 = %ld (float anticycling CC pivots)\n",ddf_statACpivots);
#endif
 ddf_WriteLPMode(f);
  ddf_WriteTimes(f,ddf_statStartTime, currenttime);
}

void ddf_WriteLPMode(FILE *f)
{
  fprintf(f, "\n* LP solver: ");
  switch (ddf_choiceLPSolverDefault) {
    case ddf_DualSimplex:
      fprintf(f, "DualSimplex\n");
      break;
    case ddf_CrissCross:
      fprintf(f, "Criss-Cross\n");
      break;
    default: break;
  }
  
  fprintf(f, "* Redundancy cheking solver: ");
  switch (ddf_choiceRedcheckAlgorithm) {
    case ddf_DualSimplex:
      fprintf(f, "DualSimplex\n");
      break;
    case ddf_CrissCross:
      fprintf(f, "Criss-Cross\n");
      break;
    default: break;
  }
  
  fprintf(f, "* Lexicographic pivot: ");
  if (ddf_choiceLexicoPivotQ)  fprintf(f, " on\n"); 
  else fprintf(f, " off\n"); 

}


void ddf_WriteRunningMode(FILE *f, ddf_PolyhedraPtr poly)
{
  if (poly->child!=NULL){
    fprintf(f,"* roworder: ");
    switch (poly->child->HalfspaceOrder) {

    case ddf_MinIndex:
      fprintf(f, "minindex\n");
      break;

    case ddf_MaxIndex:
      fprintf(f, "maxindex\n");
      break;

    case ddf_MinCutoff:
      fprintf(f, "mincutoff\n");
      break;

    case ddf_MaxCutoff:
      fprintf(f, "maxcutoff\n");
    break;

    case ddf_MixCutoff:
      fprintf(f, "mixcutoff\n");
      break;

    case ddf_LexMin:
      fprintf(f, "lexmin\n");
      break;

    case ddf_LexMax:
      fprintf(f, "lexmax\n");
      break;

    case ddf_RandomRow:
      fprintf(f, "random  %d\n",poly->child->rseed);
      break;

    default: break;
    }
  }
}


void ddf_WriteCompletionStatus(FILE *f, ddf_ConePtr cone)
{
  if (cone->Iteration<cone->m && cone->CompStatus==ddf_AllFound) {
    fprintf(f,"*Computation completed at Iteration %4ld.\n", cone->Iteration);
  } 
  if (cone->CompStatus == ddf_RegionEmpty) {
    fprintf(f,"*Computation completed at Iteration %4ld because the region found empty.\n",cone->Iteration);
  }   
}

void ddf_WritePolyFile(FILE *f, ddf_PolyhedraPtr poly)
{
  ddf_WriteAmatrix(f,poly->A,poly->m,poly->d);
}


void ddf_WriteErrorMessages(FILE *f, ddf_ErrorType Error)
{
  switch (Error) {

  case ddf_DimensionTooLarge:
    fprintf(f, "*Input Error: Input matrix is too large:\n");
    fprintf(f, "*Please increase MMAX and/or NMAX in the source code and recompile.\n");
    break;

  case ddf_IFileNotFound:
    fprintf(f, "*Input Error: Specified input file does not exist.\n");
    break;

  case ddf_OFileNotOpen:
    fprintf(f, "*Output Error: Specified output file cannot be opened.\n");
    break;

  case ddf_NegativeMatrixSize:
    fprintf(f, "*Input Error: Input matrix has a negative size:\n");
    fprintf(f, "*Please check rowsize or colsize.\n");
    break;

  case ddf_ImproperInputFormat:
    fprintf(f,"*Input Error: Input format is not correct.\n");
    fprintf(f,"*Format:\n");
    fprintf(f," begin\n");
    fprintf(f,"   m   n  NumberType(real, rational or integer)\n");
    fprintf(f,"   b  -A\n");
    fprintf(f," end\n");
    break;

  case ddf_EmptyVrepresentation:
    fprintf(f, "*Input Error: V-representation is empty:\n");
    fprintf(f, "*cddlib does not accept this trivial case for which output can be any inconsistent system.\n");
    break;

  case ddf_EmptyHrepresentation:
    fprintf(f, "*Input Error: H-representation is empty.\n");
    break;

  case ddf_EmptyRepresentation:
    fprintf(f, "*Input Error: Representation is empty.\n");
    break;

  case ddf_NoLPObjective:
    fprintf(f, "*LP Error: No LP objective (max or min) is set.\n");
    break;

  case ddf_NoRealNumberSupport:
    fprintf(f, "*LP Error: The binary (with GMP Rational) does not support Real number input.\n");
    fprintf(f, "         : Use a binary compiled without -Dddf_GMPRATIONAL option.\n");
    break;

 case ddf_NotAvailForH:
    fprintf(f, "*Error: A function is called with H-rep which does not support an H-representation.\n");
    break;

 case ddf_NotAvailForV:
    fprintf(f, "*Error: A function is called with V-rep which does not support an V-representation.\n");
    break;

 case ddf_CannotHandleLinearity:
    fprintf(f, "*Error: The function called cannot handle linearity.\n");
    break;

 case ddf_RowIndexOutOfRange:
    fprintf(f, "*Error: Specified row index is out of range\n");
    break;

 case ddf_ColIndexOutOfRange:
    fprintf(f, "*Error: Specified column index is out of range\n");
    break;

 case ddf_LPCycling:
    fprintf(f, "*Error: Possibly an LP cycling occurs.  Use the Criss-Cross method.\n");
    break;
    
 case ddf_NumericallyInconsistent:
    fprintf(f, "*Error: Numerical inconsistency is found.  Use the GMP exact arithmetic.\n");
    break;
    
  case ddf_NoError:
    fprintf(f,"*No Error found.\n");
    break;
  }
}


ddf_SetFamilyPtr ddf_CopyIncidence(ddf_PolyhedraPtr poly)
{
  ddf_SetFamilyPtr F=NULL;
  ddf_bigrange k;
  ddf_rowrange i;

  if (poly->child==NULL || poly->child->CompStatus!=ddf_AllFound) goto _L99;
  if (poly->AincGenerated==ddf_FALSE) ddf_ComputeAinc(poly);
  F=ddf_CreateSetFamily(poly->n, poly->m1);
  for (i=1; i<=poly->m1; i++)
    for (k=1; k<=poly->n; k++)
      if (set_member(k,poly->Ainc[i-1])) set_addelem(F->set[k-1],i);
_L99:;
  return F;
}

ddf_SetFamilyPtr ddf_CopyInputIncidence(ddf_PolyhedraPtr poly)
{
  ddf_rowrange i;
  ddf_SetFamilyPtr F=NULL;

  if (poly->child==NULL || poly->child->CompStatus!=ddf_AllFound) goto _L99;
  if (poly->AincGenerated==ddf_FALSE) ddf_ComputeAinc(poly);
  F=ddf_CreateSetFamily(poly->m1, poly->n);
  for(i=0; i< poly->m1; i++){
    set_copy(F->set[i], poly->Ainc[i]);
  }
_L99:;
  return F;
}

ddf_SetFamilyPtr ddf_CopyAdjacency(ddf_PolyhedraPtr poly)
{
  ddf_RayPtr RayPtr1,RayPtr2;
  ddf_SetFamilyPtr F=NULL;
  long pos1, pos2;
  ddf_bigrange lstart,k,n;
  set_type linset,allset;
  ddf_boolean adj;

  if (poly->n==0 && poly->homogeneous && poly->representation==ddf_Inequality){
    n=1; /* the origin (the unique vertex) should be output. */
  } else n=poly->n;
  set_initialize(&linset, n);
  set_initialize(&allset, n);
  if (poly->child==NULL || poly->child->CompStatus!=ddf_AllFound) goto _L99;
  F=ddf_CreateSetFamily(n, n);
  if (n<=0) goto _L99;
  poly->child->LastRay->Next=NULL;
  for (RayPtr1=poly->child->FirstRay, pos1=1;RayPtr1 != NULL; 
				RayPtr1 = RayPtr1->Next, pos1++){
    for (RayPtr2=poly->child->FirstRay, pos2=1; RayPtr2 != NULL; 
					RayPtr2 = RayPtr2->Next, pos2++){
      if (RayPtr1!=RayPtr2){
        ddf_CheckAdjacency(poly->child, &RayPtr1, &RayPtr2, &adj);
        if (adj){
          set_addelem(F->set[pos1-1], pos2);
        }
      }
    }
  }
  lstart=poly->n - poly->ldim + 1;
  set_compl(allset,allset);  /* allset is set to the ground set. */
  for (k=lstart; k<=poly->n; k++){
    set_addelem(linset,k);     /* linearity set */
    set_copy(F->set[k-1],allset);  /* linearity generator is adjacent to all */
  }
  for (k=1; k<lstart; k++){
    set_uni(F->set[k-1],F->set[k-1],linset);
     /* every generator is adjacent to all linearity generators */
  }
_L99:;
  set_free(allset); set_free(linset);
  return F;
}

ddf_SetFamilyPtr ddf_CopyInputAdjacency(ddf_PolyhedraPtr poly)
{
  ddf_rowrange i,j;
  ddf_SetFamilyPtr F=NULL;

  if (poly->child==NULL || poly->child->CompStatus!=ddf_AllFound) goto _L99;
  if (poly->AincGenerated==ddf_FALSE) ddf_ComputeAinc(poly);
  F=ddf_CreateSetFamily(poly->m1, poly->m1);
  for (i=1; i<=poly->m1; i++){
    for (j=1; j<=poly->m1; j++){
      if (i!=j && ddf_InputAdjacentQ(poly, i, j)) {
        set_addelem(F->set[i-1],j);
      }
    }
  }
_L99:;
  return F;
}

ddf_MatrixPtr ddf_CopyOutput(ddf_PolyhedraPtr poly)
{
  ddf_RayPtr RayPtr;
  ddf_MatrixPtr M=NULL;
  ddf_rowrange i=0,total;
  ddf_colrange j,j1;
  myfloat b;
  ddf_RepresentationType outputrep=ddf_Inequality;
  ddf_boolean outputorigin=ddf_FALSE;

  ddf_init(b);
  total=poly->child->LinearityDim + poly->child->FeasibleRayCount;
  
  if (poly->child->d<=0 || poly->child->newcol[1]==0) total=total-1;
  if (poly->representation==ddf_Inequality) outputrep=ddf_Generator;
  if (total==0 && poly->homogeneous && poly->representation==ddf_Inequality){
    total=1;
    outputorigin=ddf_TRUE;
    /* the origin (the unique vertex) should be output. */
  }
  if (poly->child==NULL || poly->child->CompStatus!=ddf_AllFound) goto _L99;

  M=ddf_CreateMatrix(total, poly->d);
  RayPtr = poly->child->FirstRay;
  while (RayPtr != NULL) {
    if (RayPtr->feasible) {
     ddf_CopyRay(M->matrix[i], poly->d, RayPtr, outputrep, poly->child->newcol);
      i++;  /* 086 */
    }
    RayPtr = RayPtr->Next;
  }
  for (j=2; j<=poly->d; j++){
    if (poly->child->newcol[j]==0){ 
       /* original column j is dependent on others and removed for the cone */
      ddf_set(b,poly->child->Bsave[0][j-1]);
      if (outputrep==ddf_Generator && ddf_Positive(b)){
        ddf_set(M->matrix[i][0],ddf_one);  /* ddf_Normalize */
        for (j1=1; j1<poly->d; j1++) 
          ddf_div(M->matrix[i][j1],(poly->child->Bsave[j1][j-1]),b);
      } else {
        for (j1=0; j1<poly->d; j1++)
          ddf_set(M->matrix[i][j1],poly->child->Bsave[j1][j-1]);
      }
      set_addelem(M->linset, i+1);
      i++;
    }     
  }
  if (outputorigin){ 
    /* output the origin for homogeneous H-polyhedron with no rays. */
    ddf_set(M->matrix[0][0],ddf_one);
    for (j=1; j<poly->d; j++){
      ddf_set(M->matrix[0][j],ddf_purezero);
    }
  }
  ddf_MatrixIntegerFilter(M);
  if (poly->representation==ddf_Inequality)
    M->representation=ddf_Generator;
  else
    M->representation=ddf_Inequality;
_L99:;
  ddf_clear(b);
  return M;
}

ddf_MatrixPtr ddf_CopyInput(ddf_PolyhedraPtr poly)
{
  ddf_MatrixPtr M=NULL;
  ddf_rowrange i;

  M=ddf_CreateMatrix(poly->m, poly->d);
  ddf_CopyAmatrix(M->matrix, poly->A, poly->m, poly->d);
  for (i=1; i<=poly->m; i++) 
    if (poly->EqualityIndex[i]==1) set_addelem(M->linset,i);
  ddf_MatrixIntegerFilter(M);
  if (poly->representation==ddf_Generator)
    M->representation=ddf_Generator;
  else
    M->representation=ddf_Inequality;
  return M;
}

ddf_MatrixPtr ddf_CopyGenerators(ddf_PolyhedraPtr poly)
{
  ddf_MatrixPtr M=NULL;

  if (poly->representation==ddf_Generator){
    M=ddf_CopyInput(poly);
  } else {
    M=ddf_CopyOutput(poly);
  }
  return M;
}

ddf_MatrixPtr ddf_CopyInequalities(ddf_PolyhedraPtr poly)
{
  ddf_MatrixPtr M=NULL;

  if (poly->representation==ddf_Inequality){
    M=ddf_CopyInput(poly);
  } else {
    M=ddf_CopyOutput(poly);
  }
  return M;
}

/****************************************************************************************/
/*  rational number (a/b) read is taken from Vinci by Benno Bueeler and Andreas Enge    */
/****************************************************************************************/
void ddf_sread_rational_value (const char *s, myfloat value)
   /* reads a rational value from the specified string "s" and assigns it to "value"    */
   
{
   char     *numerator_s=NULL, *denominator_s=NULL, *position;
   int      sign = 1;
   double   numerator, denominator;
#if defined ddf_GMPRATIONAL
   mpz_t znum, zden;
#else
   double rvalue;
#endif
  
   /* determine the sign of the number */
   numerator_s = s;
   if (s [0] == '-')
   {  sign = -1;
      numerator_s++;
   }
   else if (s [0] == '+')
      numerator_s++;
      
   /* look for a sign '/' and eventually split the number in numerator and denominator */
   position = strchr (numerator_s, '/');
   if (position != NULL)
   {  *position = '\0'; /* terminates the numerator */
      denominator_s = position + 1;
   };

   /* determine the floating point values of numerator and denominator */
   numerator=atol (numerator_s);
  
   if (position != NULL)
   { 
     denominator=atol (denominator_s);  
   }
   else denominator = 1;

/* 
   fprintf(stderr,"\nrational_read: numerator %f\n",numerator);
   fprintf(stderr,"rational_read: denominator %f\n",denominator);
   fprintf(stderr,"rational_read: sign %d\n",sign); 
*/

#if defined ddf_GMPRATIONAL
   mpz_init_set_str(znum,numerator_s,10);
   if (sign<0) mpz_neg(znum,znum);
   mpz_init(zden); mpz_set_ui(zden,1);
   if (denominator_s!=NULL) mpz_init_set_str(zden,denominator_s,10);
   mpq_set_num(value,znum); mpq_set_den(value,zden);
   mpq_canonicalize(value);
   mpz_clear(znum); mpz_clear(zden);
   /* num=(long)sign * numerator; */
   /* den=(unsigned long) denominator; */
   /* mpq_set_si(value, num, den); */
#elif defined GMPFLOAT
   rvalue=sign * numerator/ (signed long) denominator;
   mpf_set_d(value, rvalue);
#else
   rvalue=sign * numerator/ (signed long) denominator;
   dddf_set_d(value, rvalue);
#endif
   if (ddf_debug) {
     fprintf(stderr,"rational_read: "); 
     ddf_WriteNumber(stderr,value); fprintf(stderr,"\n");
   }
}
   

void ddf_fread_rational_value (FILE *f, myfloat value)
   /* reads a rational value from the specified file "f" and assigns it to "value"      */
   
{
   char     number_s [ddf_wordlenmax];
   myfloat rational_value;
   
   ddf_init(rational_value);
   fscanf(f, "%s ", number_s);
   ddf_sread_rational_value (number_s, rational_value);
   ddf_set(value,rational_value);
   ddf_clear(rational_value);
}
   
/****************************************************************************************/


/* end of cddio.c */

