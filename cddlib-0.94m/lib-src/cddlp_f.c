/* generated automatically from cddlp.c */
/* cddlp.c:  dual simplex method c-code
   written by Komei Fukuda, fukuda@math.ethz.ch
*/

/* cddlp.c : C-Implementation of the dual simplex method for
   solving an LP: max/min  A_(m-1).x subject to  x in P, where
   P= {x :  A_i.x >= 0, i=0,...,m-2, and  x_0=1}, and
   A_i is the i-th row of an m x n matrix A.
   Please read COPYING (GNU General Public Licence) and
   the manual cddlibman.tex for detail.
*/

#include "setoper.h"
#include "cdd_f.h"
#include "splitmix64.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#if defined ddf_GMPRATIONAL
#include "cdd_f.h"
#endif

#define ddf_FALSE 0
#define ddf_TRUE 1

typedef set_type rowset;  /* set_type defined in setoper.h */
typedef set_type colset;

void ddf_CrissCrossSolve(ddf_LPPtr lp,ddf_ErrorType *);
void ddf_DualSimplexSolve(ddf_LPPtr lp,ddf_ErrorType *);
void ddf_CrissCrossMinimize(ddf_LPPtr,ddf_ErrorType *);
void ddf_CrissCrossMaximize(ddf_LPPtr,ddf_ErrorType *);
void ddf_DualSimplexMinimize(ddf_LPPtr,ddf_ErrorType *);
void ddf_DualSimplexMaximize(ddf_LPPtr,ddf_ErrorType *);
void ddf_FindLPBasis(ddf_rowrange,ddf_colrange,ddf_Amatrix,ddf_Bmatrix,ddf_rowindex,ddf_rowset,
    ddf_colindex,ddf_rowindex,ddf_rowrange,ddf_colrange,
    ddf_colrange *,int *,ddf_LPStatusType *,long *);
void ddf_FindDualFeasibleBasis(ddf_rowrange,ddf_colrange,ddf_Amatrix,ddf_Bmatrix,ddf_rowindex,
    ddf_colindex,long *,ddf_rowrange,ddf_colrange,ddf_boolean,
    ddf_colrange *,ddf_ErrorType *,ddf_LPStatusType *,long *, long maxpivots);


#ifdef ddf_GMPRATIONAL
void ddf_BasisStatus(ddf_LPPtr lpf, ddf_LPPtr lp, ddf_boolean*);
void ddf_BasisStatusMinimize(ddf_rowrange,ddf_colrange, ddf_Amatrix,ddf_Bmatrix,ddf_rowset,
    ddf_rowrange,ddf_colrange,ddf_LPStatusType,myfloat *,ddf_Arow,ddf_Arow,ddf_rowset,ddf_colindex,
    ddf_rowrange,ddf_colrange,ddf_colrange *,long *, int *, int *);
void ddf_BasisStatusMaximize(ddf_rowrange,ddf_colrange,ddf_Amatrix,ddf_Bmatrix,ddf_rowset,
    ddf_rowrange,ddf_colrange,ddf_LPStatusType,myfloat *,ddf_Arow,ddf_Arow,ddf_rowset,ddf_colindex,
    ddf_rowrange,ddf_colrange,ddf_colrange *,long *, int *, int *);
#endif

void ddf_WriteBmatrix(FILE *f,ddf_colrange d_size,ddf_Bmatrix T);
void ddf_SetNumberType(char *line,ddf_NumberType *number,ddf_ErrorType *Error);
void ddf_ComputeRowOrderVector2(ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,
    ddf_rowindex OV,ddf_RowOrderType ho,unsigned int rseed);
void ddf_SelectPreorderedNext2(ddf_rowrange m_size,ddf_colrange d_size,
    rowset excluded,ddf_rowindex OV,ddf_rowrange *hnext);
void ddf_SetSolutions(ddf_rowrange,ddf_colrange,
   ddf_Amatrix,ddf_Bmatrix,ddf_rowrange,ddf_colrange,ddf_LPStatusType,
   myfloat *,ddf_Arow,ddf_Arow,ddf_rowset,ddf_colindex,ddf_rowrange,ddf_colrange,ddf_rowindex);
   
void ddf_WriteTableau(FILE *,ddf_rowrange,ddf_colrange,ddf_Amatrix,ddf_Bmatrix,
  ddf_colindex,ddf_rowindex);

void ddf_WriteSignTableau(FILE *,ddf_rowrange,ddf_colrange,ddf_Amatrix,ddf_Bmatrix,
  ddf_colindex,ddf_rowindex);


ddf_LPSolutionPtr ddf_CopyLPSolution(ddf_LPPtr lp)
{
  ddf_LPSolutionPtr lps;
  ddf_colrange j;
  long i;

  lps=(ddf_LPSolutionPtr) calloc(1,sizeof(ddf_LPSolutionType));
  for (i=1; i<=ddf_filenamelen; i++) lps->filename[i-1]=lp->filename[i-1];
  lps->objective=lp->objective;
  lps->solver=lp->solver; 
  lps->m=lp->m;
  lps->d=lp->d;
  lps->numbtype=lp->numbtype;

  lps->LPS=lp->LPS;  /* the current solution status */
  ddf_init(lps->optvalue);
  ddf_set(lps->optvalue,lp->optvalue);  /* optimal value */
  ddf_InitializeArow(lp->d+1,&(lps->sol));
  ddf_InitializeArow(lp->d+1,&(lps->dsol));
  lps->nbindex=(long*) calloc((lp->d)+1,sizeof(long));  /* dual solution */
  for (j=0; j<=lp->d; j++){
    ddf_set(lps->sol[j],lp->sol[j]);
    ddf_set(lps->dsol[j],lp->dsol[j]);
    lps->nbindex[j]=lp->nbindex[j];
  }
  lps->pivots[0]=lp->pivots[0];
  lps->pivots[1]=lp->pivots[1];
  lps->pivots[2]=lp->pivots[2];
  lps->pivots[3]=lp->pivots[3];
  lps->pivots[4]=lp->pivots[4];
  lps->total_pivots=lp->total_pivots;

  return lps;
}


ddf_LPPtr ddf_CreateLPData(ddf_LPObjectiveType obj,
   ddf_NumberType nt,ddf_rowrange m,ddf_colrange d)
{
  ddf_LPType *lp;

  lp=(ddf_LPPtr) calloc(1,sizeof(ddf_LPType));
  lp->solver=ddf_choiceLPSolverDefault;  /* set the default lp solver */
  lp->d=d;
  lp->m=m;
  lp->numbtype=nt;
  lp->objrow=m;
  lp->rhscol=1L;
  lp->objective=ddf_LPnone;
  lp->LPS=ddf_LPSundecided;
  lp->eqnumber=0;  /* the number of equalities */

  lp->nbindex=(long*) calloc(d+1,sizeof(long));
  lp->given_nbindex=(long*) calloc(d+1,sizeof(long));
  set_initialize(&(lp->equalityset),m);  
    /* i must be in the set iff i-th row is equality . */

  lp->redcheck_extensive=ddf_FALSE; /* this is on only for RedundantExtensive */
  lp->ired=0;
  set_initialize(&(lp->redset_extra),m);  
    /* i is in the set if i-th row is newly recognized redundant (during the checking the row ired). */
  set_initialize(&(lp->redset_accum),m);  
    /* i is in the set if i-th row is recognized redundant (during the checking the row ired). */
   set_initialize(&(lp->posset_extra),m);  
    /* i is in the set if i-th row is recognized non-linearity (during the course of computation). */
 lp->lexicopivot=ddf_choiceLexicoPivotQ;  /* ddf_choice... is set in ddf_set_global_constants() */

  lp->m_alloc=lp->m+2;
  lp->d_alloc=lp->d+2;
  lp->objective=obj;
  ddf_InitializeBmatrix(lp->d_alloc,&(lp->B));
  ddf_InitializeAmatrix(lp->m_alloc,lp->d_alloc,&(lp->A));
  ddf_InitializeArow(lp->d_alloc,&(lp->sol));
  ddf_InitializeArow(lp->d_alloc,&(lp->dsol));
  ddf_init(lp->optvalue);
  return lp;
}


ddf_LPPtr ddf_Matrix2LP(ddf_MatrixPtr M, ddf_ErrorType *err)
{
  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPType *lp;
  ddf_boolean localdebug=ddf_FALSE;

  *err=ddf_NoError;
  linc=set_card(M->linset);
  m=M->rowsize+1+linc; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  d=M->colsize;
  if (localdebug) fprintf(stderr,"number of equalities = %ld\n", linc);
  
  lp=ddf_CreateLPData(M->objective, M->numbtype, m, d);
  lp->Homogeneous = ddf_TRUE;
  lp->eqnumber=linc;  /* this records the number of equations */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    if (set_member(i, M->linset)) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
                                         /* the reversed row irev is not in the equality set. */
      for (j = 1; j <= M->colsize; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-1]);
      }  /*of j*/
      if (localdebug) fprintf(stderr,"equality row %ld generates the reverse row %ld.\n",i,irev);
    }
    for (j = 1; j <= M->colsize; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-1]);
      if (j==1 && i<M->rowsize && ddf_Nonzero(M->matrix[i-1][j-1])) lp->Homogeneous = ddf_FALSE;
    }  /*of j*/
  }  /*of i*/
  for (j = 1; j <= M->colsize; j++) {
    ddf_set(lp->A[m-1][j-1],M->rowvec[j-1]);  /* objective row */
  }  /*of j*/

  return lp;
}

ddf_LPPtr ddf_Matrix2Feasibility(ddf_MatrixPtr M, ddf_ErrorType *err)
/* Load a matrix to create an LP object for feasibility.   It is 
   essentially the ddf_Matrix2LP except that the objject function
   is set to identically ZERO (maximization).
   
*/  
	 /*  094 */
{
  ddf_rowrange m, linc;
  ddf_colrange j;
  ddf_LPType *lp;

  *err=ddf_NoError;
  linc=set_card(M->linset);
  m=M->rowsize+1+linc; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  
  lp=ddf_Matrix2LP(M, err);
  lp->objective = ddf_LPmax;   /* since the objective is zero, this is not important */
  for (j = 1; j <= M->colsize; j++) {
    ddf_set(lp->A[m-1][j-1],ddf_purezero);  /* set the objective to zero. */
  }  /*of j*/

  return lp;
}

ddf_LPPtr ddf_Matrix2Feasibility2(ddf_MatrixPtr M, ddf_rowset R, ddf_rowset S, ddf_ErrorType *err)
/* Load a matrix to create an LP object for feasibility with additional equality and
   strict inequality constraints given by R and S.  There are three types of inequalities:
   
   b_r + A_r x =  0     Linearity (Equations) specified by M
   b_s + A_s x >  0     Strict Inequalities specified by row index set S
   b_t + A_t x >= 0     The rest inequalities in M
   
   Where the linearity is considered here as the union of linearity specified by
   M and the additional set R.  When S contains any linearity rows, those
   rows are considered linearity (equation).  Thus S does not overlide linearity.
   To find a feasible solution, we set an LP
   
   maximize  z
   subject to
   b_r + A_r x     =  0      all r in Linearity
   b_s + A_s x - z >= 0      for all s in S
   b_t + A_t x     >= 0      for all the rest rows t
   1           - z >= 0      to make the LP bounded.
   
   Clearly, the feasibility problem has a solution iff the LP has a positive optimal value. 
   The variable z will be the last variable x_{d+1}.
   
*/  
/*  094 */
{
  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPType *lp;
  ddf_rowset L;
  ddf_boolean localdebug=ddf_FALSE;

  *err=ddf_NoError;
  set_initialize(&L, M->rowsize);
  set_uni(L,M->linset,R);
  linc=set_card(L);
  m=M->rowsize+1+linc+1; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  d=M->colsize+1;
  if (localdebug) fprintf(stderr,"number of equalities = %ld\n", linc);
  
  lp=ddf_CreateLPData(ddf_LPmax, M->numbtype, m, d);
  lp->Homogeneous = ddf_TRUE;
  lp->eqnumber=linc;  /* this records the number of equations */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    if (set_member(i, L)) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
                                         /* the reversed row irev is not in the equality set. */
      for (j = 1; j <= M->colsize; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-1]);
      }  /*of j*/
      if (localdebug) fprintf(stderr,"equality row %ld generates the reverse row %ld.\n",i,irev);
    } else if (set_member(i, S)) {
	  ddf_set(lp->A[i-1][M->colsize],ddf_minusone);
    }
    for (j = 1; j <= M->colsize; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-1]);
      if (j==1 && i<M->rowsize && ddf_Nonzero(M->matrix[i-1][j-1])) lp->Homogeneous = ddf_FALSE;
    }  /*of j*/
  }  /*of i*/
  for (j = 1; j <= d; j++) {
    ddf_set(lp->A[m-2][j-1],ddf_purezero);  /* initialize */
  }  /*of j*/
  ddf_set(lp->A[m-2][0],ddf_one);  /* the bounding constraint. */
  ddf_set(lp->A[m-2][M->colsize],ddf_minusone);  /* the bounding constraint. */
  for (j = 1; j <= d; j++) {
    ddf_set(lp->A[m-1][j-1],ddf_purezero);  /* initialize */
  }  /*of j*/
  ddf_set(lp->A[m-1][M->colsize],ddf_one);  /* maximize  z */
  
  set_free(L);
  return lp;
}



void ddf_FreeLPData(ddf_LPPtr lp)
{
  if ((lp)!=NULL){
    ddf_clear(lp->optvalue);
    ddf_FreeArow(lp->d_alloc,lp->dsol);
    ddf_FreeArow(lp->d_alloc,lp->sol);
    ddf_FreeBmatrix(lp->d_alloc,lp->B);
    ddf_FreeAmatrix(lp->m_alloc,lp->d_alloc,lp->A);
    set_free(lp->equalityset);
    set_free(lp->redset_extra);
    set_free(lp->redset_accum);
    set_free(lp->posset_extra);
    free(lp->nbindex);
    free(lp->given_nbindex);
    free(lp);
  }
}

void ddf_FreeLPSolution(ddf_LPSolutionPtr lps)
{
  if (lps!=NULL){
    free(lps->nbindex);
    ddf_FreeArow(lps->d+1,lps->dsol);
    ddf_FreeArow(lps->d+1,lps->sol);
    ddf_clear(lps->optvalue);
    
    free(lps);
  }
}

int ddf_LPReverseRow(ddf_LPPtr lp, ddf_rowrange i)
{
  ddf_colrange j;
  int success=0;

  if (i>=1 && i<=lp->m){
    lp->LPS=ddf_LPSundecided;
    for (j=1; j<=lp->d; j++) {
      ddf_neg(lp->A[i-1][j-1],lp->A[i-1][j-1]);
      /* negating the i-th constraint of A */
    }
    success=1;
  }
  return success;
}

int ddf_LPReplaceRow(ddf_LPPtr lp, ddf_rowrange i, ddf_Arow a)
{
  ddf_colrange j;
  int success=0;

  if (i>=1 && i<=lp->m){
    lp->LPS=ddf_LPSundecided;
    for (j=1; j<=lp->d; j++) {
      ddf_set(lp->A[i-1][j-1],a[j-1]);
      /* replacing the i-th constraint by a */
    }
    success=1;
  }
  return success;
}

ddf_Arow ddf_LPCopyRow(ddf_LPPtr lp, ddf_rowrange i)
{
  ddf_colrange j;
  ddf_Arow a;

  if (i>=1 && i<=lp->m){
    ddf_InitializeArow(lp->d, &a);
    for (j=1; j<=lp->d; j++) {
      ddf_set(a[j-1],lp->A[i-1][j-1]);
      /* copying the i-th row to a */
    }
  }
  return a;
}


void ddf_SetNumberType(char *line,ddf_NumberType *number,ddf_ErrorType *Error)
{
  if (strncmp(line,"integer",7)==0) {
    *number = ddf_Integer;
    return;
  }
  else if (strncmp(line,"rational",8)==0) {
    *number = ddf_Rational;
    return;
  }
  else if (strncmp(line,"real",4)==0) {
    *number = ddf_Real;
    return;
  }
  else { 
    *number=ddf_Unknown;
    *Error=ddf_ImproperInputFormat;
  }
}


void ddf_WriteTableau(FILE *f,ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,ddf_Bmatrix T,
  ddf_colindex nbindex,ddf_rowindex bflag)
/* Write the tableau  A.T   */
{
  ddf_colrange j;
  ddf_rowrange i;
  myfloat x;
  
  ddf_init(x);
  fprintf(f," %ld  %ld  real\n",m_size,d_size);
  fprintf(f,"          |");
  for (j=1; j<= d_size; j++) {
    fprintf(f," %ld",nbindex[j]);
  } fprintf(f,"\n");
  for (j=1; j<= d_size+1; j++) {
    fprintf(f," ----");
  } fprintf(f,"\n");
  for (i=1; i<= m_size; i++) {
    fprintf(f," %3ld(%3ld) |",i,bflag[i]);  
    for (j=1; j<= d_size; j++) {
      ddf_TableauEntry(&x,m_size,d_size,A,T,i,j);
      ddf_WriteNumber(f,x);
    }
    fprintf(f,"\n");
  }
  fprintf(f,"end\n");
  ddf_clear(x);
}

void ddf_WriteSignTableau(FILE *f,ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,ddf_Bmatrix T,
  ddf_colindex nbindex,ddf_rowindex bflag)
/* Write the sign tableau  A.T   */
{
  ddf_colrange j;
  ddf_rowrange i;
  myfloat x;
  
  ddf_init(x);
  fprintf(f," %ld  %ld  real\n",m_size,d_size);
  fprintf(f,"          |");
  for (j=1; j<= d_size; j++) {
    fprintf(f,"%3ld",nbindex[j]);
  } fprintf(f,"\n  ------- | ");
  for (j=1; j<= d_size; j++) {
    fprintf(f,"---");
  } fprintf(f,"\n");
  for (i=1; i<= m_size; i++) {
    fprintf(f," %3ld(%3ld) |",i,bflag[i]);
    for (j=1; j<= d_size; j++) {
      ddf_TableauEntry(&x,m_size,d_size,A,T,i,j);
      if (ddf_Positive(x)) fprintf(f, "  +");
      else if (ddf_Negative(x)) fprintf(f, "  -");
        else  fprintf(f, "  0");
    }
    fprintf(f,"\n");
  }
  fprintf(f,"end\n");
  ddf_clear(x);
}

void ddf_WriteSignTableau2(FILE *f,ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,ddf_Bmatrix T,
  ddf_colindex nbindex_ref, ddf_colindex nbindex,ddf_rowindex bflag)
/* Write the sign tableau  A.T  and the reference basis */
{
  ddf_colrange j;
  ddf_rowrange i;
  myfloat x;
  
  ddf_init(x);
  fprintf(f," %ld  %ld  real\n",m_size,d_size);
  fprintf(f,"          |");
  for (j=1; j<= d_size; j++) fprintf(f,"%3ld",nbindex_ref[j]);
  fprintf(f,"\n          |");
  for (j=1; j<= d_size; j++) {
    fprintf(f,"%3ld",nbindex[j]);
  } fprintf(f,"\n  ------- | ");
  for (j=1; j<= d_size; j++) {
    fprintf(f,"---");
  } fprintf(f,"\n");
  for (i=1; i<= m_size; i++) {
    fprintf(f," %3ld(%3ld) |",i,bflag[i]);
    for (j=1; j<= d_size; j++) {
      ddf_TableauEntry(&x,m_size,d_size,A,T,i,j);
      if (ddf_Positive(x)) fprintf(f, "  +");
      else if (ddf_Negative(x)) fprintf(f, "  -");
        else  fprintf(f, "  0");
    }
    fprintf(f,"\n");
  }
  fprintf(f,"end\n");
  ddf_clear(x);
}


void ddf_GetRedundancyInformation(ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,ddf_Bmatrix T,
  ddf_colindex nbindex,ddf_rowindex bflag, ddf_rowset redset)
/* Some basic variables that are forced to be nonnegative will be output.  These are
   variables whose dictionary row components are all nonnegative.   */
{
  ddf_colrange j;
  ddf_rowrange i;
  myfloat x;
  ddf_boolean red=ddf_FALSE,localdebug=ddf_FALSE;
  long numbred=0;
  
  ddf_init(x);
  for (i=1; i<= m_size; i++) {
    red=ddf_TRUE;
    for (j=1; j<= d_size; j++) {
      ddf_TableauEntry(&x,m_size,d_size,A,T,i,j);
      if (red && ddf_Negative(x)) red=ddf_FALSE;
    }
    if (bflag[i]<0 && red) {
      numbred+=1;
      set_addelem(redset,i);
    }
  }
  if (localdebug) fprintf(stderr,"\nddf_GetRedundancyInformation: %ld redundant rows over %ld\n",numbred, m_size);  
  ddf_clear(x);
}


void ddf_SelectDualSimplexPivot(ddf_rowrange m_size,ddf_colrange d_size,
    int Phase1,ddf_Amatrix A,ddf_Bmatrix T,ddf_rowindex OV,
    ddf_colindex nbindex_ref, ddf_colindex nbindex,ddf_rowindex bflag,
    ddf_rowrange objrow,ddf_colrange rhscol, ddf_boolean lexicopivot,
    ddf_rowrange *r,ddf_colrange *s,int *selected,ddf_LPStatusType *lps)
{ 
  /* selects a dual simplex pivot (*r,*s) if the current
     basis is dual feasible and not optimal. If not dual feasible,
     the procedure returns *selected=ddf_FALSE and *lps=LPSundecided.
     If Phase1=ddf_TRUE, the RHS column will be considered as the negative
     of the column of the largest variable (==m_size).  For this case, it is assumed
     that the caller used the auxiliary row (with variable m_size) to make the current
     dictionary dual feasible before calling this routine so that the nonbasic
     column for m_size corresponds to the auxiliary variable.
  */
  ddf_boolean colselected=ddf_FALSE,rowselected=ddf_FALSE,
    dualfeasible=ddf_TRUE,localdebug=ddf_FALSE;
  ddf_rowrange i,iref;
  ddf_colrange j,k;
  myfloat val,valn, minval,rat,minrat;
  static ddf_Arow rcost;
  static ddf_colrange d_last=0;
  static ddf_colset tieset,stieset;  /* store the column indices with tie */

  ddf_init(val); ddf_init(valn); ddf_init(minval); ddf_init(rat); ddf_init(minrat);
  if (d_last<d_size) {
    if (d_last>0) {
      for (j=1; j<=d_last; j++){ ddf_clear(rcost[j-1]);}
      free(rcost);
      set_free(tieset);
      set_free(stieset);
    }
    rcost=(myfloat*) calloc(d_size,sizeof(myfloat));
    for (j=1; j<=d_size; j++){ ddf_init(rcost[j-1]);}
    set_initialize(&tieset,d_size);
    set_initialize(&stieset,d_size);
    d_last=d_size;
  }

  *r=0; *s=0;
  *selected=ddf_FALSE;
  *lps=ddf_LPSundecided;
  for (j=1; j<=d_size; j++){
    if (j!=rhscol){
      ddf_TableauEntry(&(rcost[j-1]),m_size,d_size,A,T,objrow,j);
      if (ddf_Positive(rcost[j-1])) { 
        dualfeasible=ddf_FALSE;
      }
    }
  }
  if (dualfeasible){
    while ((*lps==ddf_LPSundecided) && (!rowselected) && (!colselected)) {
      for (i=1; i<=m_size; i++) {
        if (i!=objrow && bflag[i]==-1) {  /* i is a basic variable */
          if (Phase1){
            ddf_TableauEntry(&val, m_size,d_size,A,T,i,bflag[m_size]);
            ddf_neg(val,val);
            /* for dual Phase I.  The RHS (dual objective) is the negative of the auxiliary variable column. */
          } 
          else {ddf_TableauEntry(&val,m_size,d_size,A,T,i,rhscol);}
          if (ddf_Smaller(val,minval)) {
            *r=i;
            ddf_set(minval,val);
          }
        }
      }
      if (ddf_Nonnegative(minval)) {
        *lps=ddf_Optimal;
      }
      else {
        rowselected=ddf_TRUE;
        set_emptyset(tieset);
        for (j=1; j<=d_size; j++){
          ddf_TableauEntry(&val,m_size,d_size,A,T,*r,j);
          if (j!=rhscol && ddf_Positive(val)) {
            ddf_div(rat,rcost[j-1],val);
            ddf_neg(rat,rat);
            if (*s==0 || ddf_Smaller(rat,minrat)){
              ddf_set(minrat,rat);
              *s=j;
              set_emptyset(tieset);
              set_addelem(tieset, j);
            } else if (ddf_Equal(rat,minrat)){
              set_addelem(tieset,j);
            }
          }
        }
        if (*s>0) {
          if (!lexicopivot || set_card(tieset)==1){
            colselected=ddf_TRUE; *selected=ddf_TRUE;
          } else { /* lexicographic rule with respect to the given reference cobasis.  */
            if (localdebug) {printf("Tie occurred at:"); set_write(tieset); printf("\n");
              ddf_WriteTableau(stderr,m_size,d_size,A,T,nbindex,bflag);
            }
            *s=0;
            k=2; /* k runs through the column indices except RHS. */
            do {
              iref=nbindex_ref[k];  /* iref runs though the reference basic indices */
              if (iref>0) {
                j=bflag[iref];
                if (j>0) {
                  if (set_member(j,tieset) && set_card(tieset)==1) {
                    *s=j;
                     colselected=ddf_TRUE;
                  } else {
                    set_delelem(tieset, j);
                    /* iref is cobasic, and the corresponding col is not the pivot column except it is the last one. */
                  }
                } else {
                  *s=0;
                  for (j=1; j<=d_size; j++){
                    if (set_member(j,tieset)) {
                      ddf_TableauEntry(&val,m_size,d_size,A,T,*r,j);
                      ddf_TableauEntry(&valn,m_size,d_size,A,T,iref,j);
                      if (j!=rhscol && ddf_Positive(val)) {
                        ddf_div(rat,valn,val);
                        if (*s==0 || ddf_Smaller(rat,minrat)){
                          ddf_set(minrat,rat);
                          *s=j;
                          set_emptyset(stieset);
                          set_addelem(stieset, j);
                        } else if (ddf_Equal(rat,minrat)){
                          set_addelem(stieset,j);
                        }
                      }
                    }
                  }
                  set_copy(tieset,stieset);              
                  if (set_card(tieset)==1) colselected=ddf_TRUE;
                }
              }
              k+=1;
            } while (!colselected && k<=d_size);
            *selected=ddf_TRUE;
          }
        } else *lps=ddf_Inconsistent;
      }
    } /* end of while */
  }
  if (localdebug) {
     if (Phase1) fprintf(stderr,"Phase 1 : select %ld,%ld\n",*r,*s);
     else fprintf(stderr,"Phase 2 : select %ld,%ld\n",*r,*s);
  }
  ddf_clear(val); ddf_clear(valn); ddf_clear(minval); ddf_clear(rat); ddf_clear(minrat);
}

void ddf_TableauEntry(myfloat *x,ddf_rowrange m_size, ddf_colrange d_size, ddf_Amatrix X, ddf_Bmatrix T,
				ddf_rowrange r, ddf_colrange s)
/* Compute the (r,s) entry of X.T   */
{
  ddf_colrange j;
  myfloat temp;

  ddf_init(temp);
  ddf_set(*x,ddf_purezero);
  for (j=0; j< d_size; j++) {
    ddf_mul(temp,X[r-1][j], T[j][s-1]);
    ddf_add(*x, *x, temp);
  }
  ddf_clear(temp);
}

void ddf_SelectPivot2(ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,ddf_Bmatrix T,
            ddf_RowOrderType roworder,ddf_rowindex ordervec, rowset equalityset,
            ddf_rowrange rowmax,rowset NopivotRow,
            colset NopivotCol,ddf_rowrange *r,ddf_colrange *s,
            ddf_boolean *selected)
/* Select a position (*r,*s) in the matrix A.T such that (A.T)[*r][*s] is nonzero
   The choice is feasible, i.e., not on NopivotRow and NopivotCol, and
   best with respect to the specified roworder 
 */
{
  int stop;
  ddf_rowrange i,rtemp;
  rowset rowexcluded;
  myfloat Xtemp;
  ddf_boolean localdebug=ddf_FALSE;

  stop = ddf_FALSE;
  localdebug=ddf_debug;
  ddf_init(Xtemp);
  set_initialize(&rowexcluded,m_size);
  set_copy(rowexcluded,NopivotRow);
  for (i=rowmax+1;i<=m_size;i++) {
    set_addelem(rowexcluded,i);   /* cannot pivot on any row > rmax */
  }
  *selected = ddf_FALSE;
  do {
    rtemp=0; i=1;
    while (i<=m_size && rtemp==0) {  /* equalityset vars have highest priorities */
      if (set_member(i,equalityset) && !set_member(i,rowexcluded)){
        if (localdebug) fprintf(stderr,"marked set %ld chosen as a candidate\n",i);
        rtemp=i;
      }
      i++;
    }
    if (rtemp==0) ddf_SelectPreorderedNext2(m_size,d_size,rowexcluded,ordervec,&rtemp);;
    if (rtemp>=1) {
      *r=rtemp;
      *s=1;
      while (*s <= d_size && !*selected) {
        ddf_TableauEntry(&Xtemp,m_size,d_size,A,T,*r,*s);
        if (!set_member(*s,NopivotCol) && ddf_Nonzero(Xtemp)) {
          *selected = ddf_TRUE;
          stop = ddf_TRUE;
        } else {
          (*s)++;
        }
      }
      if (!*selected) {
        set_addelem(rowexcluded,rtemp);
      }
    }
    else {
      *r = 0;
      *s = 0;
      stop = ddf_TRUE;
    }
  } while (!stop);
  set_free(rowexcluded); ddf_clear(Xtemp);
}

void ddf_GaussianColumnPivot(ddf_rowrange m_size, ddf_colrange d_size, 
    ddf_Amatrix X, ddf_Bmatrix T, ddf_rowrange r, ddf_colrange s)
/* Update the Transformation matrix T with the pivot operation on (r,s) 
   This procedure performs a implicit pivot operation on the matrix X by
   updating the dual basis inverse  T.
 */
{
  ddf_colrange j, j1;
  myfloat Xtemp0, Xtemp1, Xtemp;
  static ddf_Arow Rtemp;
  static ddf_colrange last_d=0;

  ddf_init(Xtemp0); ddf_init(Xtemp1); ddf_init(Xtemp);
  if (last_d!=d_size){
    if (last_d>0) {
      for (j=1; j<=last_d; j++) ddf_clear(Rtemp[j-1]);
      free(Rtemp);
    }
    Rtemp=(myfloat*)calloc(d_size,sizeof(myfloat));
    for (j=1; j<=d_size; j++) ddf_init(Rtemp[j-1]);
    last_d=d_size;
  }

  for (j=1; j<=d_size; j++) {
    ddf_TableauEntry(&(Rtemp[j-1]), m_size, d_size, X, T, r,j);
  }
  ddf_set(Xtemp0,Rtemp[s-1]);
  for (j = 1; j <= d_size; j++) {
    if (j != s) {
      ddf_div(Xtemp,Rtemp[j-1],Xtemp0);
      ddf_set(Xtemp1,ddf_purezero);
      for (j1 = 1; j1 <= d_size; j1++){
        ddf_mul(Xtemp1,Xtemp,T[j1-1][s - 1]);
        ddf_sub(T[j1-1][j-1],T[j1-1][j-1],Xtemp1);
 /*     T[j1-1][j-1] -= T[j1-1][s - 1] * Xtemp / Xtemp0;  */
      }
    }
  }
  for (j = 1; j <= d_size; j++)
    ddf_div(T[j-1][s - 1],T[j-1][s - 1],Xtemp0);

  ddf_clear(Xtemp0); ddf_clear(Xtemp1); ddf_clear(Xtemp);
}

void ddf_GaussianColumnPivot2(ddf_rowrange m_size,ddf_colrange d_size,
    ddf_Amatrix A,ddf_Bmatrix T,ddf_colindex nbindex,ddf_rowindex bflag,ddf_rowrange r,ddf_colrange s)
/* Update the Transformation matrix T with the pivot operation on (r,s) 
   This procedure performs a implicit pivot operation on the matrix A by
   updating the dual basis inverse  T.
 */
{
  int localdebug=ddf_FALSE;
  long entering;

  if (ddf_debug) localdebug=ddf_debug;
  ddf_GaussianColumnPivot(m_size,d_size,A,T,r,s);
  entering=nbindex[s];
  bflag[r]=s;     /* the nonbasic variable r corresponds to column s */
  nbindex[s]=r;   /* the nonbasic variable on s column is r */

  if (entering>0) bflag[entering]=-1;
     /* original variables have negative index and should not affect the row index */

  if (localdebug) {
    fprintf(stderr,"ddf_GaussianColumnPivot2\n");
    fprintf(stderr," pivot: (leaving, entering) = (%ld, %ld)\n", r,entering);
    fprintf(stderr, " bflag[%ld] is set to %ld\n", r, s);
  }
}


void ddf_ResetTableau(ddf_rowrange m_size,ddf_colrange d_size,ddf_Bmatrix T,
    ddf_colindex nbindex,ddf_rowindex bflag,ddf_rowrange objrow,ddf_colrange rhscol)
{
  ddf_rowrange i;
  ddf_colrange j;
  
  /* Initialize T and nbindex */
  for (j=1; j<=d_size; j++) nbindex[j]=-j;
  nbindex[rhscol]=0; 
    /* RHS is already in nonbasis and is considered to be associated
       with the zero-th row of input. */
  ddf_SetToIdentity(d_size,T);
  
  /* Set the bflag according to nbindex */
  for (i=1; i<=m_size; i++) bflag[i]=-1;  
    /* all basic variables have index -1 */
  bflag[objrow]= 0; 
    /* bflag of the objective variable is 0,
       different from other basic variables which have -1 */
  for (j=1; j<=d_size; j++) if (nbindex[j]>0) bflag[nbindex[j]]=j;
    /* bflag of a nonbasic variable is its column number */

}

void ddf_SelectCrissCrossPivot(ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,ddf_Bmatrix T,
    ddf_rowindex bflag,ddf_rowrange objrow,ddf_colrange rhscol,
    ddf_rowrange *r,ddf_colrange *s,
    int *selected,ddf_LPStatusType *lps)
{
  int colselected=ddf_FALSE,rowselected=ddf_FALSE;
  ddf_rowrange i;
  myfloat val;
  
  ddf_init(val);
  *selected=ddf_FALSE;
  *lps=ddf_LPSundecided;
  while ((*lps==ddf_LPSundecided) && (!rowselected) && (!colselected)) {
    for (i=1; i<=m_size; i++) {
      if (i!=objrow && bflag[i]==-1) {  /* i is a basic variable */
        ddf_TableauEntry(&val,m_size,d_size,A,T,i,rhscol);
        if (ddf_Negative(val)) {
          rowselected=ddf_TRUE;
          *r=i;
          break;
        }
      }
      else if (bflag[i] >0) { /* i is nonbasic variable */
        ddf_TableauEntry(&val,m_size,d_size,A,T,objrow,bflag[i]);
        if (ddf_Positive(val)) {
          colselected=ddf_TRUE;
          *s=bflag[i];
          break;
        }
      }
    }
    if  ((!rowselected) && (!colselected)) {
      *lps=ddf_Optimal;
      return;
    }
    else if (rowselected) {
     for (i=1; i<=m_size; i++) {
       if (bflag[i] >0) { /* i is nonbasic variable */
          ddf_TableauEntry(&val,m_size,d_size,A,T,*r,bflag[i]);
          if (ddf_Positive(val)) {
            colselected=ddf_TRUE;
            *s=bflag[i];
            *selected=ddf_TRUE;
            break;
          }
        }
      }
    }
    else if (colselected) {
      for (i=1; i<=m_size; i++) {
        if (i!=objrow && bflag[i]==-1) {  /* i is a basic variable */
          ddf_TableauEntry(&val,m_size,d_size,A,T,i,*s);
          if (ddf_Negative(val)) {
            rowselected=ddf_TRUE;
            *r=i;
            *selected=ddf_TRUE;
            break;
          }
        }
      }
    }
    if (!rowselected) {
      *lps=ddf_DualInconsistent;
    }
    else if (!colselected) {
      *lps=ddf_Inconsistent;
    }
  }
  ddf_clear(val);
}

void ddf_CrissCrossSolve(ddf_LPPtr lp, ddf_ErrorType *err)
{
  switch (lp->objective) {
    case ddf_LPmax:
         ddf_CrissCrossMaximize(lp,err);
      break;
      
    case ddf_LPmin:
         ddf_CrissCrossMinimize(lp,err);
      break;

    case ddf_LPnone: *err=ddf_NoLPObjective; break;
  }

}

void ddf_DualSimplexSolve(ddf_LPPtr lp, ddf_ErrorType *err)
{
  switch (lp->objective) {
    case ddf_LPmax:
         ddf_DualSimplexMaximize(lp,err);
      break;
      
    case ddf_LPmin:
         ddf_DualSimplexMinimize(lp,err);
      break;

    case ddf_LPnone: *err=ddf_NoLPObjective; break;
  }
}

#ifdef ddf_GMPRATIONAL

ddf_LPStatusType LPSf2LPS(ddf_LPStatusType lpsf)
{
   ddf_LPStatusType lps=ddf_LPSundecided;

   switch (lpsf) {
   case ddf_LPSundecided: lps=ddf_LPSundecided; break;
   case ddf_Optimal: lps=ddf_Optimal; break;
   case ddf_Inconsistent: lps=ddf_Inconsistent; break; 
   case ddf_DualInconsistent: lps=ddf_DualInconsistent; break;
   case ddf_StrucInconsistent: lps=ddf_StrucInconsistent; break; 
   case ddf_StrucDualInconsistent: lps=ddf_StrucDualInconsistent; break;
   case ddf_Unbounded: lps=ddf_Unbounded; break;
   case ddf_DualUnbounded: lps=ddf_DualUnbounded; break;
   }
   return lps;
}


void ddf_BasisStatus(ddf_LPPtr lpf, ddf_LPPtr lp, ddf_boolean *LPScorrect)
{
  int i;
  ddf_colrange se, j;
  ddf_boolean basisfound; 
 
  switch (lp->objective) {
    case ddf_LPmax:
      ddf_BasisStatusMaximize(lp->m,lp->d,lp->A,lp->B,lp->equalityset,lp->objrow,lp->rhscol,
           lpf->LPS,&(lp->optvalue),lp->sol,lp->dsol,lp->posset_extra,lpf->nbindex,lpf->re,lpf->se,&se,lp->pivots, 
           &basisfound, LPScorrect);
      if (*LPScorrect) {
         /* printf("BasisStatus Check: the current basis is verified with GMP\n"); */
         lp->LPS=LPSf2LPS(lpf->LPS);
         lp->re=lpf->re;
         lp->se=se;
         for (j=1; j<=lp->d; j++) lp->nbindex[j]=lpf->nbindex[j]; 
      }
      for (i=1; i<=5; i++) lp->pivots[i-1]+=lpf->pivots[i-1]; 
      break;
    case ddf_LPmin:
      ddf_BasisStatusMinimize(lp->m,lp->d,lp->A,lp->B,lp->equalityset,lp->objrow,lp->rhscol,
           lpf->LPS,&(lp->optvalue),lp->sol,lp->dsol,lp->posset_extra,lpf->nbindex,lpf->re,lpf->se,&se,lp->pivots, 
           &basisfound, LPScorrect);
      if (*LPScorrect) {
         /* printf("BasisStatus Check: the current basis is verified with GMP\n"); */
         lp->LPS=LPSf2LPS(lpf->LPS);
         lp->re=lpf->re;
         lp->se=se;
         for (j=1; j<=lp->d; j++) lp->nbindex[j]=lpf->nbindex[j]; 
      }
      for (i=1; i<=5; i++) lp->pivots[i-1]+=lpf->pivots[i-1]; 
      break;
    case ddf_LPnone:  break;
   }      
}
#endif

void ddf_FindLPBasis(ddf_rowrange m_size,ddf_colrange d_size,
    ddf_Amatrix A, ddf_Bmatrix T,ddf_rowindex OV,ddf_rowset equalityset, ddf_colindex nbindex,
    ddf_rowindex bflag,ddf_rowrange objrow,ddf_colrange rhscol,
    ddf_colrange *cs,int *found,ddf_LPStatusType *lps,long *pivot_no)
{ 
  /* Find a LP basis using Gaussian pivots.
     If the problem has an LP basis,
     the procedure returns *found=ddf_TRUE,*lps=LPSundecided and an LP basis.
     If the constraint matrix A (excluding the rhs and objective) is not
     column independent, there are two cases.  If the dependency gives a dual
     inconsistency, this returns *found=ddf_FALSE, *lps=ddf_StrucDualInconsistent and 
     the evidence column *s.  Otherwise, this returns *found=ddf_TRUE, 
     *lps=LPSundecided and an LP basis of size less than d_size.  Columns j
     that do not belong to the basis (i.e. cannot be chosen as pivot because
     they are all zero) will be indicated in nbindex vector: nbindex[j] will
     be negative and set to -j.
  */
  int chosen,stop;
  long pivots_p0=0,rank;
  colset ColSelected;
  rowset RowSelected;
  myfloat val;

  ddf_rowrange r;
  ddf_colrange j,s;

  ddf_init(val);
  *found=ddf_FALSE; *cs=0; rank=0;
  stop=ddf_FALSE;
  *lps=ddf_LPSundecided;

  set_initialize(&RowSelected,m_size);
  set_initialize(&ColSelected,d_size);
  set_addelem(RowSelected,objrow);
  set_addelem(ColSelected,rhscol);

  stop=ddf_FALSE;
  do {   /* Find a LP basis */
    ddf_SelectPivot2(m_size,d_size,A,T,ddf_MinIndex,OV,equalityset,
      m_size,RowSelected,ColSelected,&r,&s,&chosen);
    if (chosen) {
      set_addelem(RowSelected,r);
      set_addelem(ColSelected,s);
      ddf_GaussianColumnPivot2(m_size,d_size,A,T,nbindex,bflag,r,s);
      pivots_p0++;
      rank++;
    } else {
      for (j=1;j<=d_size  && *lps==ddf_LPSundecided; j++) {
        if (j!=rhscol && nbindex[j]<0){
          ddf_TableauEntry(&val,m_size,d_size,A,T,objrow,j);
          if (ddf_Nonzero(val)){  /* dual inconsistent */
            *lps=ddf_StrucDualInconsistent;
            *cs=j;
            /* dual inconsistent because the nonzero reduced cost */
          }
        }
      }
      if (*lps==ddf_LPSundecided) *found=ddf_TRUE;  
         /* dependent columns but not dual inconsistent. */
      stop=ddf_TRUE;
    }
    /* printf("d_size=%ld, rank=%ld\n",d_size,rank); */
    if (rank==d_size-1) {
      stop = ddf_TRUE;
      *found=ddf_TRUE;
    }
  } while (!stop);

  *pivot_no=pivots_p0;
  ddf_statBApivots+=pivots_p0;
  set_free(RowSelected);
  set_free(ColSelected);
  ddf_clear(val);
}


void ddf_FindLPBasis2(ddf_rowrange m_size,ddf_colrange d_size,
    ddf_Amatrix A, ddf_Bmatrix T,ddf_rowindex OV,ddf_rowset equalityset, ddf_colindex nbindex,
    ddf_rowindex bflag,ddf_rowrange objrow,ddf_colrange rhscol,
    ddf_colrange *cs,int *found,long *pivot_no)
{ 
  /* Similar to ddf_FindLPBasis but it is much simpler.  This tries to recompute T for
  the specified basis given by nbindex.  It will return *found=ddf_FALSE if the specified
  basis is not a basis.
  */
  int chosen,stop;
  long pivots_p0=0,rank;
  ddf_colset ColSelected,DependentCols;
  ddf_rowset RowSelected, NopivotRow;
  myfloat val;
  ddf_boolean localdebug=ddf_FALSE;

  ddf_rowrange r,negcount=0;
  ddf_colrange j,s;

  ddf_init(val);
  *found=ddf_FALSE; *cs=0; rank=0;

  set_initialize(&RowSelected,m_size);
  set_initialize(&DependentCols,d_size);
  set_initialize(&ColSelected,d_size);
  set_initialize(&NopivotRow,m_size);
  set_addelem(RowSelected,objrow);
  set_addelem(ColSelected,rhscol);
  set_compl(NopivotRow, NopivotRow);  /* set NopivotRow to be the groundset */
  
  for (j=2; j<=d_size; j++) 
    if (nbindex[j]>0) 
       set_delelem(NopivotRow, nbindex[j]);
    else if (nbindex[j]<0){ 
       negcount++;       
       set_addelem(DependentCols, -nbindex[j]); 
       set_addelem(ColSelected, -nbindex[j]); 
    }
 
  set_uni(RowSelected, RowSelected, NopivotRow);  /* RowSelected is the set of rows not allowed to poviot on */

  stop=ddf_FALSE;
  do {   /* Find a LP basis */
    ddf_SelectPivot2(m_size,d_size,A,T,ddf_MinIndex,OV,equalityset, m_size,RowSelected,ColSelected,&r,&s,&chosen);
    if (chosen) {
      set_addelem(RowSelected,r);
      set_addelem(ColSelected,s);

      ddf_GaussianColumnPivot2(m_size,d_size,A,T,nbindex,bflag,r,s);
      if (localdebug && m_size <=10){
        ddf_WriteBmatrix(stderr,d_size,T);
        ddf_WriteTableau(stderr,m_size,d_size,A,T,nbindex,bflag);
      }
      pivots_p0++;
      rank++;
    } else{
      *found=ddf_FALSE;   /* cannot pivot on any of the spacified positions. */
      stop=ddf_TRUE;
    }
    if (rank==d_size-1-negcount) {
      if (negcount){
        /* Now it tries to pivot on rows that are supposed to be dependent. */ 
        set_diff(ColSelected, ColSelected, DependentCols); 
        ddf_SelectPivot2(m_size,d_size,A,T,ddf_MinIndex,OV,equalityset, m_size,RowSelected,ColSelected,&r,&s,&chosen);
        if (chosen) *found=ddf_FALSE;  /* not supposed to be independent */
        else *found=ddf_TRUE;
        if (localdebug){
          printf("Try to check the dependent cols:");
          set_write(DependentCols);
          if (chosen) printf("They are not dependent.  Can still pivot on (%ld, %ld)\n",r, s);
          else printf("They are indeed dependent.\n");
        }
      } else {
        *found=ddf_TRUE;
     }   
     stop = ddf_TRUE;
    }
  } while (!stop);

  for (j=1; j<=d_size; j++) if (nbindex[j]>0) bflag[nbindex[j]]=j;
  *pivot_no=pivots_p0;
  set_free(RowSelected);
  set_free(ColSelected);
  set_free(NopivotRow);
  set_free(DependentCols);
  ddf_clear(val);
}

void ddf_FindDualFeasibleBasis(ddf_rowrange m_size,ddf_colrange d_size,
    ddf_Amatrix A,ddf_Bmatrix T,ddf_rowindex OV,
    ddf_colindex nbindex,ddf_rowindex bflag,ddf_rowrange objrow,ddf_colrange rhscol, ddf_boolean lexicopivot,
    ddf_colrange *s,ddf_ErrorType *err,ddf_LPStatusType *lps,long *pivot_no, long maxpivots)
{ 
  /* Find a dual feasible basis using Phase I of Dual Simplex method.
     If the problem is dual feasible,
     the procedure returns *err=NoError, *lps=LPSundecided and a dual feasible
     basis.   If the problem is dual infeasible, this returns
     *err=NoError, *lps=DualInconsistent and the evidence column *s.
     Caution: matrix A must have at least one extra row:  the row space A[m_size] must
     have been allocated.
  */
  ddf_boolean phase1,dualfeasible=ddf_TRUE;
  ddf_boolean localdebug=ddf_FALSE,chosen,stop;
  ddf_LPStatusType LPSphase1;
  long pivots_p1=0;
  ddf_rowrange i,r_val;
  ddf_colrange j,l,ms=0,s_val,local_m_size;
  myfloat x,val,maxcost,axvalue,maxratio;
  static ddf_colrange d_last=0;
  static ddf_Arow rcost;
  static ddf_colindex nbindex_ref; /* to be used to store the initial feasible basis for lexico rule */

  myfloat scaling,svalue;  /* random scaling myfloat value */
  myfloat minval;

  if (ddf_debug) localdebug=ddf_debug;
  ddf_init(x); ddf_init(val); ddf_init(scaling); ddf_init(svalue);  ddf_init(axvalue);
  ddf_init(maxcost);  ddf_set(maxcost,ddf_minuszero);
  ddf_init(maxratio);  ddf_set(maxratio,ddf_minuszero);
  if (d_last<d_size) {
    if (d_last>0) {
      for (j=1; j<=d_last; j++){ ddf_clear(rcost[j-1]);}
      free(rcost);
      free(nbindex_ref);
    }
    rcost=(myfloat*) calloc(d_size,sizeof(myfloat));
    nbindex_ref=(long*) calloc(d_size+1,sizeof(long));
    for (j=1; j<=d_size; j++){ ddf_init(rcost[j-1]);}
    d_last=d_size;
  }

  *err=ddf_NoError; *lps=ddf_LPSundecided; *s=0;
  local_m_size=m_size+1;  /* increase m_size by 1 */

  ms=0;  /* ms will be the index of column which has the largest reduced cost */
  for (j=1; j<=d_size; j++){
    if (j!=rhscol){
      ddf_TableauEntry(&(rcost[j-1]),local_m_size,d_size,A,T,objrow,j);
      if (ddf_Larger(rcost[j-1],maxcost)) {ddf_set(maxcost,rcost[j-1]); ms = j;}
    }
  }
  if (ddf_Positive(maxcost)) dualfeasible=ddf_FALSE;

  if (!dualfeasible){
    for (j=1; j<=d_size; j++){
      ddf_set(A[local_m_size-1][j-1], ddf_purezero);
      for (l=1; l<=d_size; l++){
        if (nbindex[l]>0) {
          ddf_set_si(scaling,l+10);
          ddf_mul(svalue,A[nbindex[l]-1][j-1],scaling); 
          ddf_sub(A[local_m_size-1][j-1],A[local_m_size-1][j-1],svalue); 
          /* To make the auxiliary row (0,-11,-12,...,-d-10).
             It is likely to be better than  (0, -1, -1, ..., -1)
             to avoid a degenerate LP. */
        }
      }
    }
    
    if (localdebug){
      fprintf(stderr,"\nddf_FindDualFeasibleBasis: curruent basis is not dual feasible.\n");
      fprintf(stderr,"because of the column %ld assoc. with var %ld   dual cost =",
       ms,nbindex[ms]);
      ddf_WriteNumber(stderr, maxcost);
      if (localdebug) {
        if (m_size <=100 && d_size <=30){
          printf("\nddf_FindDualFeasibleBasis: the starting dictionary.\n");
          ddf_WriteTableau(stdout,m_size+1,d_size,A,T,nbindex,bflag);
        }
      }
    }
    
    ms=0; 
     /* Ratio Test: ms will be now the index of column which has the largest reduced cost 
        over the auxiliary row entry */
    for (j=1; j<=d_size; j++){
      if ((j!=rhscol) && ddf_Positive(rcost[j-1])){
        ddf_TableauEntry(&axvalue,local_m_size,d_size,A,T,local_m_size,j);
        if (ddf_Nonnegative(axvalue)) {
          *err=ddf_NumericallyInconsistent; 
           /* This should not happen as they are set negative above.  Quit the phase I.*/
          if (localdebug) fprintf(stderr,"ddf_FindDualFeasibleBasis: Numerical Inconsistency detected.\n");
          goto _L99;
        }
        ddf_neg(axvalue,axvalue);
        ddf_div(axvalue,rcost[j-1],axvalue);  /* axvalue is the negative of ratio that is to be maximized. */
        if (ddf_Larger(axvalue,maxratio)) {
          ddf_set(maxratio,axvalue); 
          ms = j;
        }
      }
    }

    if (ms==0) {
      *err=ddf_NumericallyInconsistent; /* This should not happen. Quit the phase I.*/
      if (localdebug) fprintf(stderr,"ddf_FindDualFeasibleBasis: Numerical Inconsistency detected.\n");
      goto _L99;
    }

    /* Pivot on (local_m_size,ms) so that the dual basic solution becomes feasible */
    ddf_GaussianColumnPivot2(local_m_size,d_size,A,T,nbindex,bflag,local_m_size,ms);
    pivots_p1=pivots_p1+1;
    if (localdebug) {
      printf("\nddf_FindDualFeasibleBasis: Pivot on %ld %ld.\n",local_m_size,ms);
    }

  for (j=1; j<=d_size; j++) nbindex_ref[j]=nbindex[j];
     /* set the reference basis to be the current feasible basis. */
  if (localdebug){
    fprintf(stderr, "Store the current feasible basis:");
    for (j=1; j<=d_size; j++) fprintf(stderr, " %ld", nbindex_ref[j]);
    fprintf(stderr, "\n");   
    if (m_size <=100 && d_size <=30)
      ddf_WriteSignTableau2(stdout,m_size+1,d_size,A,T,nbindex_ref,nbindex,bflag);
  }

    phase1=ddf_TRUE; stop=ddf_FALSE;
    do {   /* Dual Simplex Phase I */
      chosen=ddf_FALSE; LPSphase1=ddf_LPSundecided;
      if (pivots_p1>maxpivots) {
        *err=ddf_LPCycling;
        fprintf(stderr,"max number %ld of pivots performed in Phase I. Switch to the anticycling phase.\n", maxpivots);
        goto _L99;  /* failure due to max no. of pivots performed */
      }
      ddf_SelectDualSimplexPivot(local_m_size,d_size,phase1,A,T,OV,nbindex_ref,nbindex,bflag,
        objrow,rhscol,lexicopivot,&r_val,&s_val,&chosen,&LPSphase1);
      if (!chosen) {
        /* The current dictionary is terminal.  There are two cases:
           ddf_TableauEntry(local_m_size,d_size,A,T,objrow,ms) is negative or zero.
           The first case implies dual infeasible,
           and the latter implies dual feasible but local_m_size is still in nonbasis.
           We must pivot in the auxiliary variable local_m_size. 
        */
        ddf_TableauEntry(&x,local_m_size,d_size,A,T,objrow,ms);
        if (ddf_Negative(x)){
          *err=ddf_NoError; *lps=ddf_DualInconsistent;  *s=ms;
        }
        if (localdebug) {
          fprintf(stderr,"\nddf_FindDualFeasibleBasis: the auxiliary variable was forced to enter the basis (# pivots = %ld).\n",pivots_p1);
          fprintf(stderr," -- objrow %ld, ms %ld entry: ",objrow,ms);
          ddf_WriteNumber(stderr, x); fprintf(stderr,"\n");
          if (ddf_Negative(x)){
            fprintf(stderr,"->The basis is dual inconsistent. Terminate.\n");
          } else {
            fprintf(stderr,"->The basis is feasible. Go to phase II.\n");
          }
        }

        ddf_init(minval);
        r_val=0;
        for (i=1; i<=local_m_size; i++){
          if (bflag[i]<0) { 
             /* i is basic and not the objective variable */
            ddf_TableauEntry(&val,local_m_size,d_size,A,T,i,ms);  /* auxiliary column*/
            if (ddf_Smaller(val, minval)) {
              r_val=i;
              ddf_set(minval,val);
            }
          }
        }
        ddf_clear(minval);
        
        if (r_val==0) {
          *err=ddf_NumericallyInconsistent; /* This should not happen. Quit the phase I.*/
          if (localdebug) fprintf(stderr,"ddf_FindDualFeasibleBasis: Numerical Inconsistency detected (r_val is 0).\n");
          goto _L99;
        }

        ddf_GaussianColumnPivot2(local_m_size,d_size,A,T,nbindex,bflag,r_val,ms);
        pivots_p1=pivots_p1+1;
        if (localdebug) {
          printf("\nddf_FindDualFeasibleBasis: make the %ld-th pivot on %ld  %ld to force the auxiliary variable to enter the basis.\n",pivots_p1,r_val,ms);
          if (m_size <=100 && d_size <=30)
            ddf_WriteSignTableau2(stdout,m_size+1,d_size,A,T,nbindex_ref,nbindex,bflag);
        }

        stop=ddf_TRUE;

      } else {
        ddf_GaussianColumnPivot2(local_m_size,d_size,A,T,nbindex,bflag,r_val,s_val);  
        pivots_p1=pivots_p1+1;
        if (localdebug) {
          printf("\nddf_FindDualFeasibleBasis: make a %ld-th pivot on %ld  %ld\n",pivots_p1,r_val,s_val);
          if (m_size <=100 && d_size <=30)
            ddf_WriteSignTableau2(stdout,local_m_size,d_size,A,T,nbindex_ref,nbindex,bflag);
        }


        if (bflag[local_m_size]<0) {
          stop=ddf_TRUE; 
          if (localdebug) 
            fprintf(stderr,"\nDualSimplex Phase I: the auxiliary variable entered the basis (# pivots = %ld).\nGo to phase II\n",pivots_p1);
        }
      }
    } while(!stop);
  }
_L99:
  *pivot_no=pivots_p1;
  ddf_statDS1pivots+=pivots_p1;
  ddf_clear(x); ddf_clear(val); ddf_clear(maxcost); ddf_clear(maxratio);
  ddf_clear(scaling); ddf_clear(svalue); ddf_clear(axvalue);
}

void ddf_DualSimplexMinimize(ddf_LPPtr lp,ddf_ErrorType *err)
{
   ddf_colrange j;

   *err=ddf_NoError;
   for (j=1; j<=lp->d; j++)
     ddf_neg(lp->A[lp->objrow-1][j-1],lp->A[lp->objrow-1][j-1]);
   ddf_DualSimplexMaximize(lp,err);
   ddf_neg(lp->optvalue,lp->optvalue);
   for (j=1; j<=lp->d; j++){
     if (lp->LPS!=ddf_Inconsistent) {
	   /* Inconsistent certificate stays valid for minimization, 0.94e */
	   ddf_neg(lp->dsol[j-1],lp->dsol[j-1]);
	 }
     ddf_neg(lp->A[lp->objrow-1][j-1],lp->A[lp->objrow-1][j-1]);
   }
}

void ddf_DualSimplexMaximize(ddf_LPPtr lp,ddf_ErrorType *err)
/* 
When LP is inconsistent then lp->re returns the evidence row.
When LP is dual-inconsistent then lp->se returns the evidence column.
*/
{
  int stop,chosen,phase1,found;
  long pivots_ds=0,pivots_p0=0,pivots_p1=0,pivots_pc=0,maxpivots,maxpivfactor=20;
  ddf_boolean localdebug=ddf_FALSE,localdebug1=ddf_FALSE;

#if !defined ddf_GMPRATIONAL
  long maxccpivots,maxccpivfactor=100; 
    /* criss-cross should not cycle, but with floating-point arithmetics, it happens
       (very rarely).  Jorg Rambau reported such an LP, in August 2003.  Thanks Jorg!
    */
#endif

  ddf_rowrange i,r;
  ddf_colrange j,s;
  static ddf_rowindex bflag;
  static long mlast=0,nlast=0;
  static ddf_rowindex OrderVector;  /* the permutation vector to store a preordered row indeces */
  static ddf_colindex nbindex_ref; /* to be used to store the initial feasible basis for lexico rule */

  double redpercent=0,redpercent_prev=0,redgain=0;
  unsigned int rseed=1;
  
  /* *err=ddf_NoError; */
  if (ddf_debug) localdebug=ddf_debug;
  set_emptyset(lp->redset_extra);
  for (i=0; i<= 4; i++) lp->pivots[i]=0;
  maxpivots=maxpivfactor*lp->d;  /* maximum pivots to be performed before cc pivot is applied. */
#if !defined ddf_GMPRATIONAL
  maxccpivots=maxccpivfactor*lp->d;  /* maximum pivots to be performed with emergency cc pivots. */
#endif
  if (mlast!=lp->m || nlast!=lp->d){
     if (mlast>0) { /* called previously with different lp->m */
       free(OrderVector);
       free(bflag);
       free(nbindex_ref);
     }
     OrderVector=(long *)calloc(lp->m+1,sizeof(*OrderVector));
     bflag=(long *) calloc(lp->m+2,sizeof(*bflag));  /* one more element for an auxiliary variable  */
     nbindex_ref=(long*) calloc(lp->d+1,sizeof(long));
     mlast=lp->m;nlast=lp->d;
  }
  /* Initializing control variables. */
  ddf_ComputeRowOrderVector2(lp->m,lp->d,lp->A,OrderVector,ddf_MinIndex,rseed);

  lp->re=0; lp->se=0;
  
  ddf_ResetTableau(lp->m,lp->d,lp->B,lp->nbindex,bflag,lp->objrow,lp->rhscol);
   
  ddf_FindLPBasis(lp->m,lp->d,lp->A,lp->B,OrderVector,lp->equalityset,lp->nbindex,bflag,
      lp->objrow,lp->rhscol,&s,&found,&(lp->LPS),&pivots_p0);
  lp->pivots[0]=pivots_p0;

  if (!found){
     lp->se=s;
     goto _L99;
     /* No LP basis is found, and thus Inconsistent.  
     Output the evidence column. */
  }

  ddf_FindDualFeasibleBasis(lp->m,lp->d,lp->A,lp->B,OrderVector,lp->nbindex,bflag,
      lp->objrow,lp->rhscol,lp->lexicopivot,&s, err,&(lp->LPS),&pivots_p1, maxpivots);
  lp->pivots[1]=pivots_p1;

  for (j=1; j<=lp->d; j++) nbindex_ref[j]=lp->nbindex[j];
     /* set the reference basis to be the current feasible basis. */
  if (localdebug){
    fprintf(stderr, "ddf_DualSimplexMaximize: Store the current feasible basis:");
    for (j=1; j<=lp->d; j++) fprintf(stderr, " %ld", nbindex_ref[j]);
    fprintf(stderr, "\n");
    if (lp->m <=100 && lp->d <=30)
      ddf_WriteSignTableau2(stdout,lp->m+1,lp->d,lp->A,lp->B,nbindex_ref,lp->nbindex,bflag); 
  }
  
  if (*err==ddf_LPCycling || *err==ddf_NumericallyInconsistent){
    if (localdebug) fprintf(stderr, "Phase I failed and thus switch to the Criss-Cross method\n");
    ddf_CrissCrossMaximize(lp,err);
    return;
  }

  if (lp->LPS==ddf_DualInconsistent){
     lp->se=s;
     goto _L99;
     /* No dual feasible basis is found, and thus DualInconsistent.  
     Output the evidence column. */
  }

  /* Dual Simplex Method */
  stop=ddf_FALSE;
  do {
    chosen=ddf_FALSE; lp->LPS=ddf_LPSundecided; phase1=ddf_FALSE;
    if (pivots_ds<maxpivots) {
      ddf_SelectDualSimplexPivot(lp->m,lp->d,phase1,lp->A,lp->B,OrderVector,nbindex_ref,lp->nbindex,bflag,
        lp->objrow,lp->rhscol,lp->lexicopivot,&r,&s,&chosen,&(lp->LPS));
    }
    if (chosen) {
      pivots_ds=pivots_ds+1;
      if (lp->redcheck_extensive) {
        ddf_GetRedundancyInformation(lp->m,lp->d,lp->A,lp->B,lp->nbindex, bflag, lp->redset_extra);
        set_uni(lp->redset_accum, lp->redset_accum,lp->redset_extra);
        redpercent=100*(double)set_card(lp->redset_extra)/(double)lp->m;
        redgain=redpercent-redpercent_prev;
        redpercent_prev=redpercent;
        if (localdebug1){
          fprintf(stderr,"\nddf_DualSimplexMaximize: Phase II pivot %ld on (%ld, %ld).\n",pivots_ds,r,s);
          fprintf(stderr,"  redundancy %f percent: redset size = %ld\n",redpercent,set_card(lp->redset_extra));
        }
      }
    }
    if (!chosen && lp->LPS==ddf_LPSundecided) {  
      if (localdebug1){
         fprintf(stderr,"Warning: an emergency CC pivot in Phase II is performed\n");
         /* In principle this should not be executed because we already have dual feasibility
            attained and dual simplex pivot should have been chosen.  This might occur
            under floating point computation, or the case of cycling.
         */
      if (localdebug && lp->m <=100 && lp->d <=30){
          fprintf(stderr,"\nddf_DualSimplexMaximize: The current dictionary.\n");
          ddf_WriteSignTableau2(stdout,lp->m,lp->d,lp->A,lp->B,nbindex_ref,lp->nbindex,bflag);
      }
    }

#if !defined ddf_GMPRATIONAL
      if (pivots_pc>maxccpivots) {
        *err=ddf_LPCycling;
        stop=ddf_TRUE;
        goto _L99;
      }
#endif
      
      ddf_SelectCrissCrossPivot(lp->m,lp->d,lp->A,lp->B,bflag,
        lp->objrow,lp->rhscol,&r,&s,&chosen,&(lp->LPS));
      if (chosen) pivots_pc=pivots_pc+1;
    }
    if (chosen) {
      ddf_GaussianColumnPivot2(lp->m,lp->d,lp->A,lp->B,lp->nbindex,bflag,r,s);
      if (localdebug && lp->m <=100 && lp->d <=30){
          fprintf(stderr,"\nddf_DualSimplexMaximize: The current dictionary.\n");
          ddf_WriteSignTableau2(stdout,lp->m,lp->d,lp->A,lp->B,nbindex_ref,lp->nbindex,bflag);
      }
    } else {
      switch (lp->LPS){
        case ddf_Inconsistent: lp->re=r;
        case ddf_DualInconsistent: lp->se=s;

        default: break;
      }
      stop=ddf_TRUE;
    }
  } while(!stop);
_L99: 
  lp->pivots[2]=pivots_ds;
  lp->pivots[3]=pivots_pc;
  ddf_statDS2pivots+=pivots_ds;
  ddf_statACpivots+=pivots_pc;

  ddf_SetSolutions(lp->m,lp->d,lp->A,lp->B,lp->objrow,lp->rhscol,lp->LPS,&(lp->optvalue),lp->sol,lp->dsol,lp->posset_extra,lp->nbindex,lp->re,lp->se,bflag);

}



void ddf_CrissCrossMinimize(ddf_LPPtr lp,ddf_ErrorType *err)
{
   ddf_colrange j;

   *err=ddf_NoError;
   for (j=1; j<=lp->d; j++)
     ddf_neg(lp->A[lp->objrow-1][j-1],lp->A[lp->objrow-1][j-1]);
   ddf_CrissCrossMaximize(lp,err);
   ddf_neg(lp->optvalue,lp->optvalue);
   for (j=1; j<=lp->d; j++){
     if (lp->LPS!=ddf_Inconsistent) {
	   /* Inconsistent certificate stays valid for minimization, 0.94e */
	   ddf_neg(lp->dsol[j-1],lp->dsol[j-1]);
	 }
     ddf_neg(lp->A[lp->objrow-1][j-1],lp->A[lp->objrow-1][j-1]);
   }
}

void ddf_CrissCrossMaximize(ddf_LPPtr lp,ddf_ErrorType *err)
/* 
When LP is inconsistent then lp->re returns the evidence row.
When LP is dual-inconsistent then lp->se returns the evidence column.
*/
{
  int stop,chosen,found;
  long pivots0,pivots1;
#if !defined ddf_GMPRATIONAL
  long maxpivots,maxpivfactor=1000; 
    /* criss-cross should not cycle, but with floating-point arithmetics, it happens
       (very rarely).  Jorg Rambau reported such an LP, in August 2003.  Thanks Jorg!
    */
#endif

  ddf_rowrange i,r;
  ddf_colrange s;
  static ddf_rowindex bflag;
  static long mlast=0;
  static ddf_rowindex OrderVector;  /* the permutation vector to store a preordered row indeces */
  unsigned int rseed=1;
  ddf_colindex nbtemp;

  *err=ddf_NoError;
#if !defined ddf_GMPRATIONAL
  maxpivots=maxpivfactor*lp->d;  /* maximum pivots to be performed when floating-point arithmetics is used. */
#endif
  nbtemp=(long *) calloc(lp->d+1,sizeof(long));
  for (i=0; i<= 4; i++) lp->pivots[i]=0;
  if (bflag==NULL || mlast!=lp->m){
     if (mlast!=lp->m && mlast>0) {
       free(bflag);   /* called previously with different lp->m */
       free(OrderVector);
     }
     bflag=(long *) calloc(lp->m+1,sizeof(long));
     OrderVector=(long *)calloc(lp->m+1,sizeof(long)); 
     /* initialize only for the first time or when a larger space is needed */
     
     mlast=lp->m;
  }
  /* Initializing control variables. */
  ddf_ComputeRowOrderVector2(lp->m,lp->d,lp->A,OrderVector,ddf_MinIndex,rseed);

  lp->re=0; lp->se=0; pivots1=0;

  ddf_ResetTableau(lp->m,lp->d,lp->B,lp->nbindex,bflag,lp->objrow,lp->rhscol);

  ddf_FindLPBasis(lp->m,lp->d,lp->A,lp->B,OrderVector,lp->equalityset,
      lp->nbindex,bflag,lp->objrow,lp->rhscol,&s,&found,&(lp->LPS),&pivots0);
  lp->pivots[0]+=pivots0;

  if (!found){
     lp->se=s;
     goto _L99;
     /* No LP basis is found, and thus Inconsistent.  
     Output the evidence column. */
  }

  stop=ddf_FALSE;
  do {   /* Criss-Cross Method */
#if !defined ddf_GMPRATIONAL
    if (pivots1>maxpivots) {
      *err=ddf_LPCycling;
      fprintf(stderr,"max number %ld of pivots performed by the criss-cross method. Most likely due to the floating-point arithmetics error.\n", maxpivots);
      goto _L99;  /* failure due to max no. of pivots performed */
    }
#endif

    ddf_SelectCrissCrossPivot(lp->m,lp->d,lp->A,lp->B,bflag,
       lp->objrow,lp->rhscol,&r,&s,&chosen,&(lp->LPS));
    if (chosen) {
      ddf_GaussianColumnPivot2(lp->m,lp->d,lp->A,lp->B,lp->nbindex,bflag,r,s);
      pivots1++;
    } else {
      switch (lp->LPS){
        case ddf_Inconsistent: lp->re=r;
        case ddf_DualInconsistent: lp->se=s;

        default: break;
      }
      stop=ddf_TRUE;
    }
  } while(!stop);
  
_L99:
  lp->pivots[1]+=pivots1;
  ddf_statCCpivots+=pivots1;
  ddf_SetSolutions(lp->m,lp->d,lp->A,lp->B,
   lp->objrow,lp->rhscol,lp->LPS,&(lp->optvalue),lp->sol,lp->dsol,lp->posset_extra,lp->nbindex,lp->re,lp->se,bflag);
  free(nbtemp);
}

void ddf_SetSolutions(ddf_rowrange m_size,ddf_colrange d_size,
   ddf_Amatrix A,ddf_Bmatrix T,
   ddf_rowrange objrow,ddf_colrange rhscol,ddf_LPStatusType LPS,
   myfloat *optvalue,ddf_Arow sol,ddf_Arow dsol,ddf_rowset posset, ddf_colindex nbindex,
   ddf_rowrange re,ddf_colrange se,ddf_rowindex bflag)
/* 
Assign the solution vectors to sol,dsol,*optvalue after solving
the LP.
*/
{
  ddf_rowrange i;
  ddf_colrange j;
  myfloat x,sw;
  int localdebug=ddf_FALSE;
  
  ddf_init(x); ddf_init(sw);
  if (localdebug) fprintf(stderr,"SetSolutions:\n");
  switch (LPS){
  case ddf_Optimal:
    for (j=1;j<=d_size; j++) {
      ddf_set(sol[j-1],T[j-1][rhscol-1]);
      ddf_TableauEntry(&x,m_size,d_size,A,T,objrow,j);
      ddf_neg(dsol[j-1],x);
      ddf_TableauEntry(optvalue,m_size,d_size,A,T,objrow,rhscol);
      if (localdebug) {fprintf(stderr,"dsol[%ld]= ",nbindex[j]); ddf_WriteNumber(stderr, dsol[j-1]); }
    }
    for (i=1; i<=m_size; i++) {
      if (bflag[i]==-1) {  /* i is a basic variable */
        ddf_TableauEntry(&x,m_size,d_size,A,T,i,rhscol);
        if (ddf_Positive(x)) set_addelem(posset, i);
      }
    }

    break;
  case ddf_Inconsistent:
    if (localdebug) fprintf(stderr,"SetSolutions: LP is inconsistent.\n");
    for (j=1;j<=d_size; j++) {
      ddf_set(sol[j-1],T[j-1][rhscol-1]);
      ddf_TableauEntry(&x,m_size,d_size,A,T,re,j);
      ddf_neg(dsol[j-1],x);
      if (localdebug) {fprintf(stderr,"dsol[%ld]=",nbindex[j]); 
	    ddf_WriteNumber(stderr,dsol[j-1]);
		fprintf(stderr,"\n");
	  }
    }
    break;
	
  case ddf_DualInconsistent:
    if (localdebug) printf( "SetSolutions: LP is dual inconsistent.\n");
    for (j=1;j<=d_size; j++) {
      ddf_set(sol[j-1],T[j-1][se-1]);
      ddf_TableauEntry(&x,m_size,d_size,A,T,objrow,j);
      ddf_neg(dsol[j-1],x);
      if (localdebug) {fprintf(stderr,"dsol[%ld]=",nbindex[j]);
	    ddf_WriteNumber(stderr,dsol[j-1]);
		fprintf(stderr,"\n");
	  }
    }
	break;

  case ddf_StrucDualInconsistent:
    ddf_TableauEntry(&x,m_size,d_size,A,T,objrow,se);
    if (ddf_Positive(x)) ddf_set(sw,ddf_one);
    else ddf_neg(sw,ddf_one);
    for (j=1;j<=d_size; j++) {
      ddf_mul(sol[j-1],sw,T[j-1][se-1]);
      ddf_TableauEntry(&x,m_size,d_size,A,T,objrow,j);
      ddf_neg(dsol[j-1],x);
      if (localdebug) {fprintf(stderr,"dsol[%ld]= ",nbindex[j]);ddf_WriteNumber(stderr,dsol[j-1]);}
    }
    if (localdebug) fprintf(stderr,"SetSolutions: LP is dual inconsistent.\n");
    break;

  default:break;
  }
  ddf_clear(x); ddf_clear(sw);
}


void ddf_RandomPermutation2(ddf_rowindex OV,long t,unsigned int seed)
{
  long k,j,ovj;
  double u,xk,r,rand_max=(double) UINT64_MAX;
  int localdebug=ddf_FALSE;

  srand_splitmix64(seed);
  for (j=t; j>1 ; j--) {
    r=rand_splitmix64();
    u=r/rand_max;
    xk=(double)(j*u +1);
    k=(long)xk;
    if (localdebug) fprintf(stderr,"u=%g, k=%ld, r=%g, randmax= %g\n",u,k,r,rand_max);
    ovj=OV[j];
    OV[j]=OV[k];
    OV[k]=ovj;
    if (localdebug) fprintf(stderr,"row %ld is exchanged with %ld\n",j,k); 
  }
}

void ddf_ComputeRowOrderVector2(ddf_rowrange m_size,ddf_colrange d_size,ddf_Amatrix A,
    ddf_rowindex OV,ddf_RowOrderType ho,unsigned int rseed)
{
  long i,itemp;
  
  OV[0]=0;
  switch (ho){
  case ddf_MaxIndex:
    for(i=1; i<=m_size; i++) OV[i]=m_size-i+1;
    break;

  case ddf_LexMin:
    for(i=1; i<=m_size; i++) OV[i]=i;
    ddf_QuickSort(OV,1,m_size,A,d_size);
   break;

  case ddf_LexMax:
    for(i=1; i<=m_size; i++) OV[i]=i;
    ddf_QuickSort(OV,1,m_size,A,d_size);
    for(i=1; i<=m_size/2;i++){   /* just reverse the order */
      itemp=OV[i];
      OV[i]=OV[m_size-i+1];
      OV[m_size-i+1]=itemp;
    }
    break;

  case ddf_RandomRow:
    for(i=1; i<=m_size; i++) OV[i]=i;
    if (rseed<=0) rseed=1;
    ddf_RandomPermutation2(OV,m_size,rseed);
    break;

  case ddf_MinIndex: 
    for(i=1; i<=m_size; i++) OV[i]=i;
    break;

  default: 
    for(i=1; i<=m_size; i++) OV[i]=i;
    break;
 }
}

void ddf_SelectPreorderedNext2(ddf_rowrange m_size,ddf_colrange d_size,
    rowset excluded,ddf_rowindex OV,ddf_rowrange *hnext)
{
  ddf_rowrange i,k;
  
  *hnext=0;
  for (i=1; i<=m_size && *hnext==0; i++){
    k=OV[i];
    if (!set_member(k,excluded)) *hnext=k ;
  }
}

#ifdef ddf_GMPRATIONAL

ddf_LPObjectiveType Obj2Obj(ddf_LPObjectiveType obj)
{
   ddf_LPObjectiveType objf=ddf_LPnone;

   switch (obj) {
   case ddf_LPnone: objf=ddf_LPnone; break;
   case ddf_LPmax: objf=ddf_LPmax; break;
   case ddf_LPmin: objf=ddf_LPmin; break;
   }
   return objf;
}

ddf_LPPtr ddf_LPgmp2LPf(ddf_LPPtr lp)
{
  ddf_rowrange i;
  ddf_colrange j;
  ddf_LPType *lpf;
  double val;
  ddf_boolean localdebug=ddf_FALSE;

  if (localdebug) fprintf(stderr,"Converting a GMP-LP to a float-LP.\n");
  
  lpf=ddf_CreateLPData(Obj2Obj(lp->objective), ddf_Real, lp->m, lp->d);
  lpf->Homogeneous = lp->Homogeneous;
  lpf->eqnumber=lp->eqnumber;  /* this records the number of equations */

  for (i = 1; i <= lp->m; i++) {
    if (set_member(i, lp->equalityset)) set_addelem(lpf->equalityset,i);    
          /* it is equality. Its reversed row will not be in this set */
      for (j = 1; j <= lp->d; j++) {
        val=mpq_get_d(lp->A[i-1][j-1]);
        ddf_set_d(lpf->A[i-1][j-1],val);
      }  /*of j*/
  }  /*of i*/

  return lpf;
}


#endif


ddf_boolean ddf_LPSolve(ddf_LPPtr lp,ddf_LPSolverType solver,ddf_ErrorType *err)
/* 
The current version of ddf_LPSolve that solves an LP with floating-arithmetics first
and then with the specified arithimetics if it is GMP.

When LP is inconsistent then *re returns the evidence row.
When LP is dual-inconsistent then *se returns the evidence column.
*/
{
  int i;
  ddf_boolean found=ddf_FALSE;
#ifdef ddf_GMPRATIONAL
  ddf_LPPtr lpf;
  ddf_ErrorType errf;
  ddf_boolean LPScorrect=ddf_FALSE;
  ddf_boolean localdebug=ddf_FALSE;
  if (ddf_debug) localdebug=ddf_debug;
#endif

  *err=ddf_NoError;
  lp->solver=solver;
  
   time(&lp->starttime);

#ifndef ddf_GMPRATIONAL
  switch (lp->solver) {
    case ddf_CrissCross:
      ddf_CrissCrossSolve(lp,err);
      break;
    case ddf_DualSimplex:
      ddf_DualSimplexSolve(lp,err);
      break;
  }
#else
  lpf=ddf_LPgmp2LPf(lp);
  switch (lp->solver) {
    case ddf_CrissCross:
      ddf_CrissCrossSolve(lpf,&errf);    /* First, run with double float. */
	  if (errf==ddf_NoError){   /* 094a:  fix for a bug reported by Dima Pasechnik */
        ddf_BasisStatus(lpf,lp, &LPScorrect);    /* Check the basis. */
	  } else {LPScorrect=ddf_FALSE;}
      if (!LPScorrect) {
         if (localdebug) printf("BasisStatus: the current basis is NOT verified with GMP. Rerun with GMP.\n");
         ddf_CrissCrossSolve(lp,err);  /* Rerun with GMP if fails. */
      } else {
         if (localdebug) printf("BasisStatus: the current basis is verified with GMP. The LP Solved.\n");
      }
      break;
    case ddf_DualSimplex:
      ddf_DualSimplexSolve(lpf,&errf);    /* First, run with double float. */
	  if (errf==ddf_NoError){   /* 094a:  fix for a bug reported by Dima Pasechnik */
        ddf_BasisStatus(lpf,lp, &LPScorrect);    /* Check the basis. */
	  } else {LPScorrect=ddf_FALSE;}
      if (!LPScorrect){
         if (localdebug) printf("BasisStatus: the current basis is NOT verified with GMP. Rerun with GMP.\n");
         ddf_DualSimplexSolve(lp,err);  /* Rerun with GMP if fails. */
         if (localdebug){
            printf("*total number pivots = %ld (ph0 = %ld, ph1 = %ld, ph2 = %ld, ph3 = %ld, ph4 = %ld)\n",
               lp->total_pivots,lp->pivots[0],lp->pivots[1],lp->pivots[2],lp->pivots[3],lp->pivots[4]);
            ddf_WriteLPResult(stdout, lpf, errf);
            ddf_WriteLP(stdout, lp);
         }
      } else {
         if (localdebug) printf("BasisStatus: the current basis is verified with GMP. The LP Solved.\n");
      }
      break;
  }
  ddf_FreeLPData(lpf);
#endif

  time(&lp->endtime);
  lp->total_pivots=0;
  for (i=0; i<=4; i++) lp->total_pivots+=lp->pivots[i];
  if (*err==ddf_NoError) found=ddf_TRUE;
  return found;
}


ddf_boolean ddf_LPSolve0(ddf_LPPtr lp,ddf_LPSolverType solver,ddf_ErrorType *err)
/* 
The original version of ddf_LPSolve that solves an LP with specified arithimetics.

When LP is inconsistent then *re returns the evidence row.
When LP is dual-inconsistent then *se returns the evidence column.
*/
{
  int i;
  ddf_boolean found=ddf_FALSE;

  *err=ddf_NoError;
  lp->solver=solver;
  time(&lp->starttime);

  switch (lp->solver) {
    case ddf_CrissCross:
      ddf_CrissCrossSolve(lp,err);
      break;
    case ddf_DualSimplex:
      ddf_DualSimplexSolve(lp,err);
      break;
  }

  time(&lp->endtime);
  lp->total_pivots=0;
  for (i=0; i<=4; i++) lp->total_pivots+=lp->pivots[i];
  if (*err==ddf_NoError) found=ddf_TRUE;
  return found;
}


ddf_LPPtr ddf_MakeLPforInteriorFinding(ddf_LPPtr lp)
/* Delete the objective row,
   add an extra column with -1's to the matrix A,
   add an extra row with (bceil, 0,...,0,-1),
   add an objective row with (0,...,0,1), and 
   rows & columns, and change m_size and d_size accordingly, to output new_A.
  This sets up the LP:
  maximize      x_{d+1}
  s.t.    A x + x_{d+1}  <=  b
                x_{d+1}  <=  bm * bmax,
  where bm is set to 2 by default, and bmax=max{1, b[1],...,b[m_size]}.
  Note that the equalitions (linearity) in the input lp will be ignored.
*/
{
  ddf_rowrange m;
  ddf_colrange d;
  ddf_NumberType numbtype;
  ddf_LPObjectiveType obj;
  ddf_LPType *lpnew;
  ddf_rowrange i; 
  ddf_colrange j;
  myfloat bm,bmax,bceil;
  int localdebug=ddf_FALSE;

  ddf_init(bm); ddf_init(bmax); ddf_init(bceil);
  ddf_add(bm,ddf_one,ddf_one); ddf_set(bmax,ddf_one);
  numbtype=lp->numbtype;
  m=lp->m+1;
  d=lp->d+1;
  obj=ddf_LPmax;

  lpnew=ddf_CreateLPData(obj, numbtype, m, d);

  for (i=1; i<=lp->m; i++) {
    if (ddf_Larger(lp->A[i-1][lp->rhscol-1],bmax)) 
      ddf_set(bmax,lp->A[i-1][lp->rhscol-1]);
  }
  ddf_mul(bceil,bm,bmax);
  if (localdebug) {fprintf(stderr,"bceil is set to "); ddf_WriteNumber(stderr, bceil);}
  
  for (i=1; i <= lp->m; i++) {
    for (j=1; j <= lp->d; j++) {
      ddf_set(lpnew->A[i-1][j-1],lp->A[i-1][j-1]);
    }
  }

  for (i=1;i<=lp->m; i++){
    ddf_neg(lpnew->A[i-1][lp->d],ddf_one);  /* new column with all minus one's */
  }

  for (j=1;j<=lp->d;j++){
    ddf_set(lpnew->A[m-2][j-1],ddf_purezero);   /* new row (bceil, 0,...,0,-1) */
  }
  ddf_set(lpnew->A[m-2][0],bceil);  /* new row (bceil, 0,...,0,-1) */

  for (j=1;j<= d-1;j++) {
    ddf_set(lpnew->A[m-1][j-1],ddf_purezero);  /* new obj row with (0,...,0,1) */
  }
  ddf_set(lpnew->A[m-1][d-1],ddf_one);    /* new obj row with (0,...,0,1) */
 
  if (localdebug) ddf_WriteAmatrix(stderr, lp->A, lp->m, lp->d);
  if (localdebug) ddf_WriteAmatrix(stderr, lpnew->A, lpnew->m, lpnew->d);
  ddf_clear(bm); ddf_clear(bmax); ddf_clear(bceil);

  return lpnew;
}


void ddf_WriteLPResult(FILE *f,ddf_LPPtr lp,ddf_ErrorType err)
{
  long j;

  fprintf(f,"* cdd LP solver result\n");
  
  if (err!=ddf_NoError) {
    ddf_WriteErrorMessages(f,err);
    goto _L99;
  }

  ddf_WriteProgramDescription(f);

  fprintf(f,"* #constraints = %ld\n",lp->m-1);
  fprintf(f,"* #variables   = %ld\n",lp->d-1);

  switch (lp->solver) {
    case ddf_DualSimplex:
      fprintf(f,"* Algorithm: dual simplex algorithm\n");break; 
    case ddf_CrissCross:
      fprintf(f,"* Algorithm: criss-cross method\n");break;
  }

  switch (lp->objective) {
    case ddf_LPmax:
      fprintf(f,"* maximization is chosen\n");break; 
    case ddf_LPmin:
      fprintf(f,"* minimization is chosen\n");break;
    case ddf_LPnone:
      fprintf(f,"* no objective type (max or min) is chosen\n");break;
  }
  
  if (lp->objective==ddf_LPmax||lp->objective==ddf_LPmin){
    fprintf(f,"* Objective function is\n");  
    for (j=0; j<lp->d; j++){
      if (j>0 && ddf_Nonnegative(lp->A[lp->objrow-1][j]) ) fprintf(f," +");
      if (j>0 && (j % 5) == 0) fprintf(f,"\n");
      ddf_WriteNumber(f,lp->A[lp->objrow-1][j]);
      if (j>0) fprintf(f," X[%3ld]",j);
    }
    fprintf(f,"\n");
  }

  switch (lp->LPS){
  case ddf_Optimal:
    fprintf(f,"* LP status: a dual pair (x,y) of optimal solutions found.\n");
    fprintf(f,"begin\n");
    fprintf(f,"  primal_solution\n");
    for (j=1; j<lp->d; j++) {
      fprintf(f,"  %3ld : ",j);
      ddf_WriteNumber(f,lp->sol[j]);
      fprintf(f,"\n");
    }
    fprintf(f,"  dual_solution\n");
    for (j=1; j<lp->d; j++){
      if (lp->nbindex[j+1]>0) {
        fprintf(f,"  %3ld : ",lp->nbindex[j+1]);
        ddf_WriteNumber(f,lp->dsol[j]); fprintf(f,"\n");
      }
    }
    fprintf(f,"  optimal_value : "); ddf_WriteNumber(f,lp->optvalue);
    fprintf(f,"\nend\n");
    break;

  case ddf_Inconsistent:
    fprintf(f,"* LP status: LP is inconsistent.\n");
    fprintf(f,"* The positive combination of original inequalities with\n");
    fprintf(f,"* the following coefficients will prove the inconsistency.\n");
    fprintf(f,"begin\n");
    fprintf(f,"  dual_direction\n");
    fprintf(f,"  %3ld : ",lp->re);
    ddf_WriteNumber(f,ddf_one);  fprintf(f,"\n");
    for (j=1; j<lp->d; j++){
      if (lp->nbindex[j+1]>0) {
        fprintf(f,"  %3ld : ",lp->nbindex[j+1]);
        ddf_WriteNumber(f,lp->dsol[j]); fprintf(f,"\n");
      }
    }
    fprintf(f,"end\n");
    break;

  case ddf_DualInconsistent: case ddf_StrucDualInconsistent:
    fprintf(f,"* LP status: LP is dual inconsistent.\n");
    fprintf(f,"* The linear combination of columns with\n");
    fprintf(f,"* the following coefficients will prove the dual inconsistency.\n");
    fprintf(f,"* (It is also an unbounded direction for the primal LP.)\n");
    fprintf(f,"begin\n");
    fprintf(f,"  primal_direction\n");
    for (j=1; j<lp->d; j++) {
      fprintf(f,"  %3ld : ",j);
      ddf_WriteNumber(f,lp->sol[j]);
      fprintf(f,"\n");
    }
    fprintf(f,"end\n");
    break;

  default:
    break;
  }
  fprintf(f,"* number of pivot operations = %ld (ph0 = %ld, ph1 = %ld, ph2 = %ld, ph3 = %ld, ph4 = %ld)\n",lp->total_pivots,lp->pivots[0],lp->pivots[1],lp->pivots[2],lp->pivots[3],lp->pivots[4]);
  ddf_WriteLPTimes(f, lp);
_L99:;
}

ddf_LPPtr ddf_CreateLP_H_ImplicitLinearity(ddf_MatrixPtr M)
{
  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPPtr lp;
  ddf_boolean localdebug=ddf_FALSE;

  linc=set_card(M->linset);
  m=M->rowsize+1+linc+1; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  d=M->colsize+1;
  
  lp=ddf_CreateLPData(M->objective, M->numbtype, m, d);
  lp->Homogeneous = ddf_TRUE;
  lp->objective = ddf_LPmax;
  lp->eqnumber=linc;  /* this records the number of equations */
  lp->redcheck_extensive=ddf_FALSE;  /* this is default */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    if (set_member(i, M->linset)) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
            /* the reversed row irev is not in the equality set. */
      for (j = 1; j <= M->colsize; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-1]);
      }  /*of j*/
    } else {
      ddf_set(lp->A[i-1][d-1],ddf_minusone);  /* b_I + A_I x - 1 z >= 0  (z=x_d) */
    }
    for (j = 1; j <= M->colsize; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-1]);
      if (j==1 && i<M->rowsize && ddf_Nonzero(M->matrix[i-1][j-1])) lp->Homogeneous = ddf_FALSE;
    }  /*of j*/
  }  /*of i*/
  ddf_set(lp->A[m-2][0],ddf_one);  ddf_set(lp->A[m-2][d-1],ddf_minusone);
      /* make the LP bounded.  */
  
  ddf_set(lp->A[m-1][d-1],ddf_one);
      /* objective is to maximize z.  */

  if (localdebug) {
    fprintf(stderr,"ddf_CreateLP_H_ImplicitLinearity: an new lp is\n");
    ddf_WriteLP(stderr,lp);
  }

  return lp;
}

ddf_LPPtr ddf_CreateLP_V_ImplicitLinearity(ddf_MatrixPtr M)
{
  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPPtr lp;
  ddf_boolean localdebug=ddf_FALSE;

  linc=set_card(M->linset);
  m=M->rowsize+1+linc+1; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  d=(M->colsize)+2;  
     /* Two more columns.  This is different from the H-reprentation case */
  
/* The below must be modified for V-representation!!!  */

  lp=ddf_CreateLPData(M->objective, M->numbtype, m, d);
  lp->Homogeneous = ddf_FALSE;
  lp->objective = ddf_LPmax;
  lp->eqnumber=linc;  /* this records the number of equations */
  lp->redcheck_extensive=ddf_FALSE;  /* this is default */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    ddf_set(lp->A[i-1][0],ddf_purezero);  /* It is almost completely degerate LP */
    if (set_member(i, M->linset)) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
            /* the reversed row irev is not in the equality set. */
      for (j = 2; j <= (M->colsize)+1; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-2]);
      }  /*of j*/
      if (localdebug) fprintf(stderr,"equality row %ld generates the reverse row %ld.\n",i,irev);
    } else {
      ddf_set(lp->A[i-1][d-1],ddf_minusone);  /* b_I x_0 + A_I x - 1 z >= 0 (z=x_d) */
    }
    for (j = 2; j <= (M->colsize)+1; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-2]);
    }  /*of j*/
  }  /*of i*/
  ddf_set(lp->A[m-2][0],ddf_one);  ddf_set(lp->A[m-2][d-1],ddf_minusone);
      /* make the LP bounded.  */
  ddf_set(lp->A[m-1][d-1],ddf_one);
      /* objective is to maximize z.  */

  if (localdebug) {
    fprintf(stderr,"ddf_CreateLP_V_ImplicitLinearity: an new lp is\n");
    ddf_WriteLP(stderr,lp);
  }

  return lp;
}


ddf_LPPtr ddf_CreateLP_H_Redundancy(ddf_MatrixPtr M, ddf_rowrange itest)
{
  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPPtr lp;
  ddf_boolean localdebug=ddf_FALSE;

  linc=set_card(M->linset);
  m=M->rowsize+1+linc; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  d=M->colsize;
  
  lp=ddf_CreateLPData(M->objective, M->numbtype, m, d);
  lp->Homogeneous = ddf_TRUE;
  lp->objective = ddf_LPmin;
  lp->eqnumber=linc;  /* this records the number of equations */
  lp->redcheck_extensive=ddf_FALSE;  /* this is default */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    if (set_member(i, M->linset)) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
            /* the reversed row irev is not in the equality set. */
      for (j = 1; j <= M->colsize; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-1]);
      }  /*of j*/
      if (localdebug) fprintf(stderr,"equality row %ld generates the reverse row %ld.\n",i,irev);
    }
    for (j = 1; j <= M->colsize; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-1]);
      if (j==1 && i<M->rowsize && ddf_Nonzero(M->matrix[i-1][j-1])) lp->Homogeneous = ddf_FALSE;
    }  /*of j*/
  }  /*of i*/
  for (j = 1; j <= M->colsize; j++) {
    ddf_set(lp->A[m-1][j-1],M->matrix[itest-1][j-1]);
      /* objective is to violate the inequality in question.  */
  }  /*of j*/
  ddf_add(lp->A[itest-1][0],lp->A[itest-1][0],ddf_one); /* relax the original inequality by one */

  return lp;
}


ddf_LPPtr ddf_CreateLP_V_Redundancy(ddf_MatrixPtr M, ddf_rowrange itest)
{
  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPPtr lp;
  ddf_boolean localdebug=ddf_FALSE;

  linc=set_card(M->linset);
  m=M->rowsize+1+linc; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple. */
  d=(M->colsize)+1;  
     /* One more column.  This is different from the H-reprentation case */
  
/* The below must be modified for V-representation!!!  */

  lp=ddf_CreateLPData(M->objective, M->numbtype, m, d);
  lp->Homogeneous = ddf_FALSE;
  lp->objective = ddf_LPmin;
  lp->eqnumber=linc;  /* this records the number of equations */
  lp->redcheck_extensive=ddf_FALSE;  /* this is default */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    if (i==itest){
      ddf_set(lp->A[i-1][0],ddf_one); /* this is to make the LP bounded, ie. the min >= -1 */
    } else {
      ddf_set(lp->A[i-1][0],ddf_purezero);  /* It is almost completely degerate LP */
    }
    if (set_member(i, M->linset)) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
            /* the reversed row irev is not in the equality set. */
      for (j = 2; j <= (M->colsize)+1; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-2]);
      }  /*of j*/
      if (localdebug) fprintf(stderr,"equality row %ld generates the reverse row %ld.\n",i,irev);
    }
    for (j = 2; j <= (M->colsize)+1; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-2]);
    }  /*of j*/
  }  /*of i*/
  for (j = 2; j <= (M->colsize)+1; j++) {
    ddf_set(lp->A[m-1][j-1],M->matrix[itest-1][j-2]);
      /* objective is to violate the inequality in question.  */
  }  /*of j*/
  ddf_set(lp->A[m-1][0],ddf_purezero);   /* the constant term for the objective is zero */

  if (localdebug) ddf_WriteLP(stdout, lp);

  return lp;
}


ddf_LPPtr ddf_CreateLP_V_SRedundancy(ddf_MatrixPtr M, ddf_rowrange itest)
{
/*
     V-representation (=boundary problem)
       g* = maximize  
         1^T b_{I-itest} x_0 + 1^T A_{I-itest}    (the sum of slacks)
       subject to
         b_itest x_0     + A_itest x      =  0 (the point has to lie on the boundary)
         b_{I-itest} x_0 + A_{I-itest} x >=  0 (all nonlinearity generators in one side)
         1^T b_{I-itest} x_0 + 1^T A_{I-itest} x <=  1 (to make an LP bounded)
         b_L x_0         + A_L x = 0.  (linearity generators)
         
    The redundant row is strongly redundant if and only if g* is zero.
*/

  ddf_rowrange m, i, irev, linc;
  ddf_colrange d, j;
  ddf_LPPtr lp;
  ddf_boolean localdebug=ddf_FALSE;

  linc=set_card(M->linset);
  m=M->rowsize+1+linc+2; 
     /* We represent each equation by two inequalities.
        This is not the best way but makes the code simple.
        Two extra constraints are for the first equation and the bouding inequality.
        */
  d=(M->colsize)+1;  
     /* One more column.  This is different from the H-reprentation case */
  
/* The below must be modified for V-representation!!!  */

  lp=ddf_CreateLPData(M->objective, M->numbtype, m, d);
  lp->Homogeneous = ddf_FALSE;
  lp->objective = ddf_LPmax;
  lp->eqnumber=linc;  /* this records the number of equations */

  irev=M->rowsize; /* the first row of the linc reversed inequalities. */
  for (i = 1; i <= M->rowsize; i++) {
    if (i==itest){
      ddf_set(lp->A[i-1][0],ddf_purezero);  /* this is a half of the boundary constraint. */
    } else {
      ddf_set(lp->A[i-1][0],ddf_purezero);  /* It is almost completely degerate LP */
    }
    if (set_member(i, M->linset) || i==itest) {
      irev=irev+1;
      set_addelem(lp->equalityset,i);    /* it is equality. */
            /* the reversed row irev is not in the equality set. */
      for (j = 2; j <= (M->colsize)+1; j++) {
        ddf_neg(lp->A[irev-1][j-1],M->matrix[i-1][j-2]);
      }  /*of j*/
      if (localdebug) fprintf(stderr,"equality row %ld generates the reverse row %ld.\n",i,irev);
    }
    for (j = 2; j <= (M->colsize)+1; j++) {
      ddf_set(lp->A[i-1][j-1],M->matrix[i-1][j-2]);
      ddf_add(lp->A[m-1][j-1],lp->A[m-1][j-1],lp->A[i-1][j-1]);  /* the objective is the sum of all ineqalities */
    }  /*of j*/
  }  /*of i*/
  for (j = 2; j <= (M->colsize)+1; j++) {
    ddf_neg(lp->A[m-2][j-1],lp->A[m-1][j-1]);
      /* to make an LP bounded.  */
  }  /*of j*/
  ddf_set(lp->A[m-2][0],ddf_one);   /* the constant term for the bounding constraint is 1 */

  if (localdebug) ddf_WriteLP(stdout, lp);

  return lp;
}

ddf_boolean ddf_Redundant(ddf_MatrixPtr M, ddf_rowrange itest, ddf_Arow certificate, ddf_ErrorType *error)  
  /* 092 */
{
  /* Checks whether the row itest is redundant for the representation.
     All linearity rows are not checked and considered NONredundant. 
     This code works for both H- and V-representations.  A certificate is
     given in the case of non-redundancy, showing a solution x violating only the itest
     inequality for H-representation, a hyperplane RHS and normal (x_0, x) that
     separates the itest from the rest.  More explicitly, the LP to be setup is

     H-representation
       f* = minimize  
         b_itest     + A_itest x
       subject to
         b_itest + 1 + A_itest x     >= 0 (relaxed inequality to make an LP bounded)
         b_{I-itest} + A_{I-itest} x >= 0 (all inequalities except for itest)
         b_L         + A_L x = 0.  (linearity)

     V-representation (=separation problem)
       f* = minimize  
         b_itest x_0     + A_itest x
       subject to
         b_itest x_0     + A_itest x     >= -1 (to make an LP bounded)
         b_{I-itest} x_0 + A_{I-itest} x >=  0 (all nonlinearity generators except for itest in one side)
         b_L x_0         + A_L x = 0.  (linearity generators)
    
    Here, the input matrix is considered as (b, A), i.e. b corresponds to the first column of input
    and the row indices of input is partitioned into I and L where L is the set of linearity.
    In both cases, the itest data is nonredundant if and only if the optimal value f* is negative.
    The certificate has dimension one more for V-representation case.
  */

  ddf_colrange j;
  ddf_LPPtr lp;
  ddf_LPSolutionPtr lps;
  ddf_ErrorType err=ddf_NoError;
  ddf_boolean answer=ddf_FALSE,localdebug=ddf_FALSE;

  *error=ddf_NoError;
  if (set_member(itest, M->linset)){
    if (localdebug) printf("The %ld th row is linearity and redundancy checking is skipped.\n",itest);
    goto _L99;
  }
  
  /* Create an LP data for redundancy checking */
  if (M->representation==ddf_Generator){
    lp=ddf_CreateLP_V_Redundancy(M, itest);
  } else {
    lp=ddf_CreateLP_H_Redundancy(M, itest);
  }

  ddf_LPSolve(lp,ddf_choiceRedcheckAlgorithm,&err);
  if (err!=ddf_NoError){
    *error=err;
    goto _L999;
  } else {
    lps=ddf_CopyLPSolution(lp);

    for (j=0; j<lps->d; j++) {
      ddf_set(certificate[j], lps->sol[j]);
    }

    if (ddf_Negative(lps->optvalue)){
      answer=ddf_FALSE;
      if (localdebug) fprintf(stderr,"==> %ld th row is nonredundant.\n",itest);
    } else {
      answer=ddf_TRUE;
      if (localdebug) fprintf(stderr,"==> %ld th row is redundant.\n",itest);
    }
    ddf_FreeLPSolution(lps);
  }
  _L999:
  ddf_FreeLPData(lp);
_L99:
  return answer;
}

ddf_boolean ddf_RedundantExtensive(ddf_MatrixPtr M, ddf_rowrange itest, ddf_Arow certificate, 
ddf_rowset *redset,ddf_ErrorType *error)  
  /* 094 */
{
  /* This uses the same LP construction as ddf_Reduandant.  But, while it is checking
     the redundancy of itest, it also tries to find some other variable that are
     redundant (i.e. forced to be nonnegative).  This is expensive as it used
     the complete tableau information at each DualSimplex pivot.  The redset must
     be initialized before this function is called.
  */

  ddf_colrange j;
  ddf_LPPtr lp;
  ddf_LPSolutionPtr lps;
  ddf_ErrorType err=ddf_NoError;
  ddf_boolean answer=ddf_FALSE,localdebug=ddf_FALSE;

  *error=ddf_NoError;
  if (set_member(itest, M->linset)){
    if (localdebug) printf("The %ld th row is linearity and redundancy checking is skipped.\n",itest);
    goto _L99;
  }
  
  /* Create an LP data for redundancy checking */
  if (M->representation==ddf_Generator){
    lp=ddf_CreateLP_V_Redundancy(M, itest);
  } else {
    lp=ddf_CreateLP_H_Redundancy(M, itest);
  }
  
  lp->redcheck_extensive=ddf_TRUE;

  ddf_LPSolve0(lp,ddf_DualSimplex,&err);
  if (err!=ddf_NoError){
    *error=err;
    goto _L999;
  } else {
    set_copy(*redset,lp->redset_extra);
    set_diff(*redset, *redset, M->linset);  /* linearity set is not tested for redundancy */
    set_delelem(*redset, itest);  
    /* itest row might be redundant in the lp but this has nothing to do with its redundancy
    in the original system M.   Thus we must delete it.  */
    if (localdebug){
      fprintf(stderr, "ddf_RedundantExtensive: checking for %ld, extra redset with cardinality %ld (%ld)\n",itest,set_card(*redset),set_card(lp->redset_extra)); 
      set_fwrite(stderr, *redset); fprintf(stderr, "\n");
    }
    lps=ddf_CopyLPSolution(lp);

    for (j=0; j<lps->d; j++) {
      ddf_set(certificate[j], lps->sol[j]);
    }

    if (ddf_Negative(lps->optvalue)){
      answer=ddf_FALSE;
      if (localdebug) fprintf(stderr,"==> %ld th row is nonredundant.\n",itest);
    } else {
      answer=ddf_TRUE;
      if (localdebug) fprintf(stderr,"==> %ld th row is redundant.\n",itest);
    }
    ddf_FreeLPSolution(lps);
  }
  _L999:
  ddf_FreeLPData(lp);
_L99:
  return answer;
}

ddf_rowset ddf_RedundantRows(ddf_MatrixPtr M, ddf_ErrorType *error)  /* 092 */
{
  ddf_rowrange i,m;
  ddf_colrange d;
  ddf_rowset redset;
  ddf_MatrixPtr Mcopy;
  ddf_Arow cvec; /* certificate */  
  ddf_boolean localdebug=ddf_FALSE;

  m=M->rowsize;
  if (M->representation==ddf_Generator){
    d=(M->colsize)+1;
  } else {
    d=M->colsize;
  }
  Mcopy=ddf_MatrixCopy(M);
  ddf_InitializeArow(d,&cvec); 
  set_initialize(&redset, m);
  for (i=m; i>=1; i--) {
    if (ddf_Redundant(Mcopy, i, cvec, error)) {
      if (localdebug) printf("Iteration %ld: the row %ld is redundant.\n",m-i+1,i);
      set_addelem(redset, i);
      ddf_MatrixRowRemove(&Mcopy, i);
    } else {
      if (localdebug) printf("Iteration %ld: the row %ld is essential.\n",m-i+1, i);
    }
    if (*error!=ddf_NoError) goto _L99;
  }
_L99:
  ddf_FreeMatrix(Mcopy);
  ddf_FreeArow(d, cvec);
  return redset;
}


ddf_boolean ddf_MatrixRedundancyRemove(ddf_MatrixPtr *M, ddf_rowset *redset,ddf_rowindex *newpos, ddf_ErrorType *error) /* 094 */
{
  /* It returns the set of all redundant rows.  This should be called after all
     implicit linearity are recognized with ddf_MatrixCanonicalizeLinearity.
  */

 
  ddf_rowrange i,k,m,m1;
  ddf_colrange d;
  ddf_rowset redset1;
  ddf_rowindex newpos1;
  ddf_MatrixPtr M1=NULL;
  ddf_Arow cvec; /* certificate */ 
  ddf_boolean success=ddf_FALSE, localdebug=ddf_FALSE;

  m=(*M)->rowsize;
  set_initialize(redset, m);
  M1=ddf_MatrixSortedUniqueCopy(*M,newpos);
  for (i=1; i<=m; i++){
    if ((*newpos)[i]<=0) set_addelem(*redset,i);
    if (localdebug) printf(" %ld:%ld",i,(*newpos)[i]);
  }
  if (localdebug) printf("\n");
  
  if ((*M)->representation==ddf_Generator){
    d=((*M)->colsize)+1;
  } else {
    d=(*M)->colsize;
  }
  m1=M1->rowsize;
  if (localdebug){
    fprintf(stderr,"ddf_MatrixRedundancyRemove: By sorting, %ld rows have been removed.  The remaining has %ld rows.\n",m-m1,m1);
    /* ddf_WriteMatrix(stdout,M1);  */
  }
  ddf_InitializeArow(d,&cvec); 
  set_initialize(&redset1, M1->rowsize);
  k=1;
  do {
    if (ddf_RedundantExtensive(M1, k, cvec, &redset1,error)) {
      set_addelem(redset1, k);
      ddf_MatrixRowsRemove2(&M1,redset1,&newpos1);
      for (i=1; i<=m; i++){
        if ((*newpos)[i]>0){
          if  (set_member((*newpos)[i],redset1)){
            set_addelem(*redset,i);
            (*newpos)[i]=0;  /* now the original row i is recognized redundant and removed from M1 */
          } else {
            (*newpos)[i]=newpos1[(*newpos)[i]];  /* update the new pos vector */
          }
        }
      }
      set_free(redset1);
      set_initialize(&redset1, M1->rowsize); 
      if (localdebug) {
        printf("ddf_MatrixRedundancyRemove: the row %ld is redundant. The new matrix has %ld rows.\n", k, M1->rowsize);
        /* ddf_WriteMatrix(stderr, M1);  */
      }
      free(newpos1);
    } else {
      if (set_card(redset1)>0) {
        ddf_MatrixRowsRemove2(&M1,redset1,&newpos1);
        for (i=1; i<=m; i++){
          if ((*newpos)[i]>0){
            if  (set_member((*newpos)[i],redset1)){
              set_addelem(*redset,i);
              (*newpos)[i]=0;  /* now the original row i is recognized redundant and removed from M1 */
            } else {
              (*newpos)[i]=newpos1[(*newpos)[i]];  /* update the new pos vector */
            }
          }
        }
        set_free(redset1);
        set_initialize(&redset1, M1->rowsize);
        free(newpos1);
      }
      if (localdebug) {
        printf("ddf_MatrixRedundancyRemove: the row %ld is essential. The new matrix has %ld rows.\n", k, M1->rowsize);
        /* ddf_WriteMatrix(stderr, M1);  */
      }
      k=k+1;
    }
    if (*error!=ddf_NoError) goto _L99;
  } while  (k<=M1->rowsize);
  if (localdebug) ddf_WriteMatrix(stderr, M1);
  success=ddf_TRUE;
  
_L99:
  ddf_FreeMatrix(*M);
  *M=M1;
  ddf_FreeArow(d, cvec);
  set_free(redset1);
  return success;
}


ddf_boolean ddf_SRedundant(ddf_MatrixPtr M, ddf_rowrange itest, ddf_Arow certificate, ddf_ErrorType *error)  
  /* 093a */
{
  /* Checks whether the row itest is strongly redundant for the representation.
     A row is strongly redundant in H-representation if every point in
     the polyhedron satisfies it with strict inequality.
     A row is strongly redundant in V-representation if this point is in
     the interior of the polyhedron.
     
     All linearity rows are not checked and considered NOT strongly redundant. 
     This code works for both H- and V-representations.  A certificate is
     given in the case of non-redundancy, showing a solution x violating only the itest
     inequality for H-representation, a hyperplane RHS and normal (x_0, x) that
     separates the itest from the rest.  More explicitly, the LP to be setup is

     H-representation
       f* = minimize  
         b_itest     + A_itest x
       subject to
         b_itest + 1 + A_itest x     >= 0 (relaxed inequality to make an LP bounded)
         b_{I-itest} + A_{I-itest} x >= 0 (all inequalities except for itest)
         b_L         + A_L x = 0.  (linearity)

     V-representation (=separation problem)
       f* = minimize  
         b_itest x_0     + A_itest x
       subject to
         b_itest x_0     + A_itest x     >= -1 (to make an LP bounded)
         b_{I-itest} x_0 + A_{I-itest} x >=  0 (all nonlinearity generators except for itest in one side)
         b_L x_0         + A_L x = 0.  (linearity generators)
    
    Here, the input matrix is considered as (b, A), i.e. b corresponds to the first column of input
    and the row indices of input is partitioned into I and L where L is the set of linearity.
    In H-representation, the itest data is strongly redundant if and only if the optimal value f* is positive.
    In V-representation, the itest data is redundant if and only if the optimal value f* is zero (as the LP
    is homogeneous and the optimal value is always non-positive).  To recognize strong redundancy, one
    can set up a second LP
    
     V-representation (=boundary problem)
       g* = maximize  
         1^T b_{I-itest} x_0 + 1^T A_{I-itest}    (the sum of slacks)
       subject to
         b_itest x_0     + A_itest x      =  0 (the point has to lie on the boundary)
         b_{I-itest} x_0 + A_{I-itest} x >=  0 (all nonlinearity generators in one side)
         1^T b_{I-itest} x_0 + 1^T A_{I-itest} x <=  1 (to make an LP bounded)
         b_L x_0         + A_L x = 0.  (linearity generators)
         
    The redundant row is strongly redundant if and only if g* is zero.

    The certificate has dimension one more for V-representation case.
  */

  ddf_colrange j;
  ddf_LPPtr lp;
  ddf_LPSolutionPtr lps;
  ddf_ErrorType err=ddf_NoError;
  ddf_boolean answer=ddf_FALSE,localdebug=ddf_FALSE;

  *error=ddf_NoError;
  if (set_member(itest, M->linset)){
    if (localdebug) printf("The %ld th row is linearity and strong redundancy checking is skipped.\n",itest);
    goto _L99;
  }
  
  /* Create an LP data for redundancy checking */
  if (M->representation==ddf_Generator){
    lp=ddf_CreateLP_V_Redundancy(M, itest);
  } else {
    lp=ddf_CreateLP_H_Redundancy(M, itest);
  }

  ddf_LPSolve(lp,ddf_choiceRedcheckAlgorithm,&err);
  if (err!=ddf_NoError){
    *error=err;
    goto _L999;
  } else {
    lps=ddf_CopyLPSolution(lp);

    for (j=0; j<lps->d; j++) {
      ddf_set(certificate[j], lps->sol[j]);
    }

    if (localdebug){
      printf("Optimum value:");
      ddf_WriteNumber(stdout, lps->optvalue);
      printf("\n");
    }

    if (M->representation==ddf_Inequality){
       if (ddf_Positive(lps->optvalue)){
          answer=ddf_TRUE;
          if (localdebug) fprintf(stderr,"==> %ld th inequality is strongly redundant.\n",itest);
        } else {
          answer=ddf_FALSE;
          if (localdebug) fprintf(stderr,"==> %ld th inequality is not strongly redundant.\n",itest);
        } 
    } else {
       if (ddf_Negative(lps->optvalue)){
          answer=ddf_FALSE;
          if (localdebug) fprintf(stderr,"==> %ld th point is not strongly redundant.\n",itest);
        } else {
          /* for V-representation, we have to solve another LP */
          ddf_FreeLPData(lp);
          ddf_FreeLPSolution(lps);
          lp=ddf_CreateLP_V_SRedundancy(M, itest);
          ddf_LPSolve(lp,ddf_DualSimplex,&err);
          lps=ddf_CopyLPSolution(lp);
          if (localdebug) ddf_WriteLPResult(stdout,lp,err);
          if (ddf_Positive(lps->optvalue)){
            answer=ddf_FALSE;
            if (localdebug) fprintf(stderr,"==> %ld th point is not strongly redundant.\n",itest);
          } else {
            answer=ddf_TRUE;
            if (localdebug) fprintf(stderr,"==> %ld th point is strongly redundant.\n",itest);
          }
       }
    } 
    ddf_FreeLPSolution(lps);
  }
  _L999:
  ddf_FreeLPData(lp);
_L99:
  return answer;
}

ddf_rowset ddf_SRedundantRows(ddf_MatrixPtr M, ddf_ErrorType *error)  /* 093a */
{
  ddf_rowrange i,m;
  ddf_colrange d;
  ddf_rowset redset;
  ddf_MatrixPtr Mcopy;
  ddf_Arow cvec; /* certificate */  
  ddf_boolean localdebug=ddf_FALSE;

  m=M->rowsize;
  if (M->representation==ddf_Generator){
    d=(M->colsize)+1;
  } else {
    d=M->colsize;
  }
  Mcopy=ddf_MatrixCopy(M);
  ddf_InitializeArow(d,&cvec); 
  set_initialize(&redset, m);
  for (i=m; i>=1; i--) {
    if (ddf_SRedundant(Mcopy, i, cvec, error)) {
      if (localdebug) printf("ddf_SRedundantRows: the row %ld is strongly redundant.\n", i);
      set_addelem(redset, i);
      ddf_MatrixRowRemove(&Mcopy, i);
    } else {
      if (localdebug) printf("ddf_SRedundantRows: the row %ld is not strongly redundant.\n", i);
    }
    if (*error!=ddf_NoError) goto _L99;
  }
_L99:
  ddf_FreeMatrix(Mcopy);
  ddf_FreeArow(d, cvec);
  return redset;
}

ddf_rowset ddf_RedundantRowsViaShooting(ddf_MatrixPtr M, ddf_ErrorType *error)  /* 092 */
{
  /* 
     For H-representation only and not quite reliable,
     especially when floating-point arithmetic is used.
     Use the ordinary (slower) method ddf_RedundantRows.
  */

  ddf_rowrange i,m, ired, irow=0;
  ddf_colrange j,k,d;
  ddf_rowset redset;
  ddf_rowindex rowflag; 
    /* ith comp is negative if the ith inequality (i-1 st row) is redundant.
                   zero     if it is not decided.
                   k > 0    if it is nonredundant and assigned to the (k-1)th row of M1.
    */
  ddf_MatrixPtr M1;
  ddf_Arow shootdir, cvec=NULL;
  ddf_LPPtr lp0, lp;
  ddf_LPSolutionPtr lps; 
  ddf_ErrorType err;
  ddf_LPSolverType solver=ddf_DualSimplex; 
  ddf_boolean localdebug=ddf_FALSE;

  m=M->rowsize;
  d=M->colsize;
  M1=ddf_CreateMatrix(m,d);
  M1->rowsize=0;  /* cheat the rowsize so that smaller matrix can be stored */
  set_initialize(&redset, m);
  ddf_InitializeArow(d, &shootdir);
  ddf_InitializeArow(d, &cvec);

  rowflag=(long *)calloc(m+1, sizeof(long)); 

  /* First find some (likely) nonredundant inequalities by Interior Point Find. */
  lp0=ddf_Matrix2LP(M, &err);
  lp=ddf_MakeLPforInteriorFinding(lp0);
  ddf_FreeLPData(lp0); 
  ddf_LPSolve(lp, solver, &err);  /* Solve the LP */
  lps=ddf_CopyLPSolution(lp);

  if (ddf_Positive(lps->optvalue)){
    /* An interior point is found.  Use rayshooting to find some nonredundant
       inequalities. */
    for (j=1; j<d; j++){
      for (k=1; k<=d; k++) ddf_set(shootdir[k-1], ddf_purezero);
      ddf_set(shootdir[j], ddf_one);  /* j-th unit vector */
      ired=ddf_RayShooting(M, lps->sol, shootdir);
      if (localdebug) printf("nonredundant row %3ld found by shooting.\n", ired);
      if (ired>0 && rowflag[ired]<=0) {
        irow++;
        rowflag[ired]=irow;
        for (k=1; k<=d; k++) ddf_set(M1->matrix[irow-1][k-1], M->matrix[ired-1][k-1]); 
      }
        
      ddf_neg(shootdir[j], ddf_one);  /* negative of the j-th unit vector */
      ired=ddf_RayShooting(M, lps->sol, shootdir);
      if (localdebug) printf("nonredundant row %3ld found by shooting.\n", ired);
      if (ired>0 && rowflag[ired]<=0) {
        irow++;
        rowflag[ired]=irow;
        for (k=1; k<=d; k++) ddf_set(M1->matrix[irow-1][k-1], M->matrix[ired-1][k-1]); 
      }
    }

    M1->rowsize=irow;
    if (localdebug) {
      printf("The initial nonredundant set is:");
      for (i=1; i<=m; i++) if (rowflag[i]>0) printf(" %ld", i);
      printf("\n");
    }
    
    i=1;
    while(i<=m){
      if (rowflag[i]==0){ /* the ith inequality is not yet checked */
        if (localdebug) fprintf(stderr, "Checking redundancy of %ld th inequality\n", i);
        irow++;  M1->rowsize=irow;
        for (k=1; k<=d; k++) ddf_set(M1->matrix[irow-1][k-1], M->matrix[i-1][k-1]);
        if (!ddf_Redundant(M1, irow, cvec, &err)){
          for (k=1; k<=d; k++) ddf_sub(shootdir[k-1], cvec[k-1], lps->sol[k-1]); 
          ired=ddf_RayShooting(M, lps->sol, shootdir);
          rowflag[ired]=irow;
          for (k=1; k<=d; k++) ddf_set(M1->matrix[irow-1][k-1], M->matrix[ired-1][k-1]);
          if (localdebug) {
            fprintf(stderr, "The %ld th inequality is nonredundant for the subsystem\n", i);
            fprintf(stderr, "The nonredundancy of %ld th inequality is found by shooting.\n", ired);
          }
        } else {
          if (localdebug) fprintf(stderr, "The %ld th inequality is redundant for the subsystem and thus for the whole.\n", i);
          rowflag[i]=-1;
          set_addelem(redset, i);
          irow--;  M1->rowsize=irow;
          i++;
        }
      } else {
        i++;
      }
    } /* endwhile */
  } else {
    /* No interior point is found.  Apply the standard LP technique.  */
    if (localdebug) printf("No interior-point is found and thus the standard LP technique will be used.\n", ired);
    redset=ddf_RedundantRows(M, error);
  }

  ddf_FreeLPData(lp);
  ddf_FreeLPSolution(lps);

  M1->rowsize=m; M1->colsize=d;  /* recover the original sizes */
  ddf_FreeMatrix(M1);
  ddf_FreeArow(d, shootdir);
  ddf_FreeArow(d, cvec);
  free(rowflag);
  return redset;
}

ddf_SetFamilyPtr ddf_Matrix2Adjacency(ddf_MatrixPtr M, ddf_ErrorType *error)  /* 093 */
{
  /* This is to generate the (facet) graph of a polyheron (H) V-represented by M using LPs.
     Since it does not use the representation conversion, it should work for a large
     scale problem.
  */
  ddf_rowrange i,m;
  ddf_colrange d;
  ddf_rowset redset;
  ddf_MatrixPtr Mcopy;
  ddf_SetFamilyPtr F=NULL;

  m=M->rowsize;
  d=M->colsize;
  if (m<=0 ||d<=0) {
    *error=ddf_EmptyRepresentation;
    goto _L999;
  }
  Mcopy=ddf_MatrixCopy(M);
  F=ddf_CreateSetFamily(m, m);
  for (i=1; i<=m; i++) {
    if (!set_member(i, M->linset)){
      set_addelem(Mcopy->linset, i);
      redset=ddf_RedundantRows(Mcopy, error);  /* redset should contain all nonadjacent ones */
      set_uni(redset, redset, Mcopy->linset); /* all linearity elements should be nonadjacent */
      set_compl(F->set[i-1], redset); /* set the adjacency list of vertex i */
      set_delelem(Mcopy->linset, i);
      set_free(redset);
      if (*error!=ddf_NoError) goto _L99;
    }
  }
_L99:
  ddf_FreeMatrix(Mcopy);
_L999:
  return F;
}

ddf_SetFamilyPtr ddf_Matrix2WeakAdjacency(ddf_MatrixPtr M, ddf_ErrorType *error)  /* 093a */
{
  /* This is to generate the weak-adjacency (facet) graph of a polyheron (H) V-represented by M using LPs.
     Since it does not use the representation conversion, it should work for a large
     scale problem.
  */
  ddf_rowrange i,m;
  ddf_colrange d;
  ddf_rowset redset;
  ddf_MatrixPtr Mcopy;
  ddf_SetFamilyPtr F=NULL;

  m=M->rowsize;
  d=M->colsize;
  if (m<=0 ||d<=0) {
    *error=ddf_EmptyRepresentation;
    goto _L999;
  }
  Mcopy=ddf_MatrixCopy(M);
  F=ddf_CreateSetFamily(m, m);
  for (i=1; i<=m; i++) {
    if (!set_member(i, M->linset)){
      set_addelem(Mcopy->linset, i);
      redset=ddf_SRedundantRows(Mcopy, error);  /* redset should contain all weakly nonadjacent ones */
      set_uni(redset, redset, Mcopy->linset); /* all linearity elements should be nonadjacent */
      set_compl(F->set[i-1], redset); /* set the adjacency list of vertex i */
      set_delelem(Mcopy->linset, i);
      set_free(redset);
      if (*error!=ddf_NoError) goto _L99;
    }
  }
_L99:
  ddf_FreeMatrix(Mcopy);
_L999:
  return F;
}


ddf_boolean ddf_ImplicitLinearity(ddf_MatrixPtr M, ddf_rowrange itest, ddf_Arow certificate, ddf_ErrorType *error)  
  /* 092 */
{
  /* Checks whether the row itest is implicit linearity for the representation.
     All linearity rows are not checked and considered non implicit linearity (ddf_FALSE). 
     This code works for both H- and V-representations.  A certificate is
     given in the case of ddf_FALSE, showing a feasible solution x satisfying the itest
     strict inequality for H-representation, a hyperplane RHS and normal (x_0, x) that
     separates the itest from the rest.  More explicitly, the LP to be setup is
     the same thing as redundancy case but with maximization:

     H-representation
       f* = maximize  
         b_itest     + A_itest x
       subject to
         b_itest + 1 + A_itest x     >= 0 (relaxed inequality. This is not necessary but kept for simplicity of the code)
         b_{I-itest} + A_{I-itest} x >= 0 (all inequalities except for itest)
         b_L         + A_L x = 0.  (linearity)

     V-representation (=separation problem)
       f* = maximize  
         b_itest x_0     + A_itest x
       subject to
         b_itest x_0     + A_itest x     >= -1 (again, this is not necessary but kept for simplicity.)
         b_{I-itest} x_0 + A_{I-itest} x >=  0 (all nonlinearity generators except for itest in one side)
         b_L x_0         + A_L x = 0.  (linearity generators)
    
    Here, the input matrix is considered as (b, A), i.e. b corresponds to the first column of input
    and the row indices of input is partitioned into I and L where L is the set of linearity.
    In both cases, the itest data is implicit linearity if and only if the optimal value f* is nonpositive.
    The certificate has dimension one more for V-representation case.
  */

  ddf_colrange j;
  ddf_LPPtr lp;
  ddf_LPSolutionPtr lps;
  ddf_ErrorType err=ddf_NoError;
  ddf_boolean answer=ddf_FALSE,localdebug=ddf_FALSE;

  *error=ddf_NoError;
  if (set_member(itest, M->linset)){
    if (localdebug) printf("The %ld th row is linearity and redundancy checking is skipped.\n",itest);
    goto _L99;
  }
  
  /* Create an LP data for redundancy checking */
  if (M->representation==ddf_Generator){
    lp=ddf_CreateLP_V_Redundancy(M, itest);
  } else {
    lp=ddf_CreateLP_H_Redundancy(M, itest);
  }

  lp->objective = ddf_LPmax;  /* the lp->objective is set by CreateLP* to LPmin */
  ddf_LPSolve(lp,ddf_choiceRedcheckAlgorithm,&err);
  if (err!=ddf_NoError){
    *error=err;
    goto _L999;
  } else {
    lps=ddf_CopyLPSolution(lp);

    for (j=0; j<lps->d; j++) {
      ddf_set(certificate[j], lps->sol[j]);
    }

    if (lps->LPS==ddf_Optimal && ddf_EqualToZero(lps->optvalue)){
      answer=ddf_TRUE;
      if (localdebug) fprintf(stderr,"==> %ld th data is an implicit linearity.\n",itest);
    } else {
      answer=ddf_FALSE;
      if (localdebug) fprintf(stderr,"==> %ld th data is not an implicit linearity.\n",itest);
    }
    ddf_FreeLPSolution(lps);
  }
  _L999:
  ddf_FreeLPData(lp);
_L99:
  return answer;
}


int ddf_FreeOfImplicitLinearity(ddf_MatrixPtr M, ddf_Arow certificate, ddf_rowset *imp_linrows, ddf_ErrorType *error)  
  /* 092 */
{
  /* Checks whether the matrix M constains any implicit linearity at all.
  It returns 1 if it is free of any implicit linearity.  This means that 
  the present linearity rows define the linearity correctly.  It returns
  nonpositive values otherwise.  


     H-representation
       f* = maximize    z
       subject to
         b_I  + A_I x - 1 z >= 0 
         b_L  + A_L x = 0  (linearity)
         z <= 1.

     V-representation (=separation problem)
       f* = maximize    z
       subject to
         b_I x_0 + A_I x - 1 z >= 0 (all nonlinearity generators in one side)
         b_L x_0 + A_L x  = 0  (linearity generators)
         z <= 1.
    
    Here, the input matrix is considered as (b, A), i.e. b corresponds to the first column of input
    and the row indices of input is partitioned into I and L where L is the set of linearity.
    In both cases, any implicit linearity exists if and only if the optimal value f* is nonpositive.
    The certificate has dimension one more for V-representation case.
  */

  ddf_LPPtr lp;
  ddf_rowrange i,m;
  ddf_colrange j,d1;
  ddf_ErrorType err=ddf_NoError;
  ddf_Arow cvec; /* certificate for implicit linearity */

  int answer=0,localdebug=ddf_FALSE;

  *error=ddf_NoError;
  /* Create an LP data for redundancy checking */
  if (M->representation==ddf_Generator){
    lp=ddf_CreateLP_V_ImplicitLinearity(M);
  } else {
    lp=ddf_CreateLP_H_ImplicitLinearity(M);
  }

  ddf_LPSolve(lp,ddf_choiceRedcheckAlgorithm,&err);
  if (err!=ddf_NoError){
    *error=err;
    goto _L999;
  } else {

    for (j=0; j<lp->d; j++) {
      ddf_set(certificate[j], lp->sol[j]);
    }

    if (localdebug) ddf_WriteLPResult(stderr,lp,err);
    
    /* *posset contains a set of row indices that are recognized as nonlinearity.  */
    if (localdebug) {
      fprintf(stderr,"==> The following variables are not implicit linearity:\n");
      set_fwrite(stderr, lp->posset_extra);
      fprintf(stderr,"\n");
    }
    
    if (M->representation==ddf_Generator){
      d1=(M->colsize)+1;
    } else {
      d1=M->colsize;
    }
    m=M->rowsize;
    ddf_InitializeArow(d1,&cvec);
    set_initialize(imp_linrows,m);

    if (lp->LPS==ddf_Optimal){
      if (ddf_Positive(lp->optvalue)){
        answer=1;
        if (localdebug) fprintf(stderr,"==> The matrix has no implicit linearity.\n");
      } else if (ddf_Negative(lp->optvalue)) {
          answer=-1;
          if (localdebug) fprintf(stderr,"==> The matrix defines the trivial system.\n");
        } else {
            answer=0;
            if (localdebug) fprintf(stderr,"==> The matrix has some implicit linearity.\n");
          }
    } else {
          answer=-2;
          if (localdebug) fprintf(stderr,"==> The LP fails.\n");
    }
    if (answer==0){
      /* List the implicit linearity rows */
      for (i=m; i>=1; i--) {
        if (!set_member(i,lp->posset_extra)) {
          if (ddf_ImplicitLinearity(M, i, cvec, error)) {
            set_addelem(*imp_linrows, i);
            if (localdebug) {
              fprintf(stderr," row %ld is implicit linearity\n",i);
              fprintf(stderr,"\n");
            }
          }
          if (*error!=ddf_NoError) goto _L999;
        }
      }
    }  /* end of if (answer==0) */
    if (answer==-1) {      
      for (i=m; i>=1; i--) set_addelem(*imp_linrows, i);
    } /* all rows are considered implicit linearity */

    ddf_FreeArow(d1,cvec);
  }
  _L999:
  ddf_FreeLPData(lp);

  return answer;
}


ddf_rowset ddf_ImplicitLinearityRows(ddf_MatrixPtr M, ddf_ErrorType *error)  /* 092 */
{
  ddf_colrange d;
  ddf_rowset imp_linset;
  ddf_Arow cvec; /* certificate */
  int foi;
  ddf_boolean localdebug=ddf_FALSE;

  if (M->representation==ddf_Generator){
    d=(M->colsize)+2;
  } else {
    d=M->colsize+1;
  }

  ddf_InitializeArow(d,&cvec);
  if (localdebug) fprintf(stdout, "\nddf_ImplicitLinearityRows: Check whether the system contains any implicit linearity.\n");
  foi=ddf_FreeOfImplicitLinearity(M, cvec, &imp_linset, error);
  if (localdebug){
    switch (foi) {
      case 1:
        fprintf(stdout, "  It is free of implicit linearity.\n");
        break;
      
      case 0:
        fprintf(stdout, "  It is not free of implicit linearity.\n");
        break;

    case -1:
        fprintf(stdout, "  The input system is trivial (i.e. the empty H-polytope or the V-rep of the whole space.\n");
        break;
    
    default:
        fprintf(stdout, "  The LP was not solved correctly.\n");
        break;
    
    }
  }
  
  if (localdebug){
    fprintf(stderr, "  Implicit linearity rows are:\n");
    set_fwrite(stderr,imp_linset);
    fprintf(stderr, "\n");  
  }
  ddf_FreeArow(d, cvec);
  return imp_linset;
}

ddf_boolean ddf_MatrixCanonicalizeLinearity(ddf_MatrixPtr *M, ddf_rowset *impl_linset,ddf_rowindex *newpos, 
ddf_ErrorType *error) /* 094 */
{
/* This is to recongnize all implicit linearities, and put all linearities at the top of
   the matrix.    All implicit linearities will be returned by *impl_linset.
*/
  ddf_rowrange rank;
  ddf_rowset linrows,ignoredrows,basisrows;
  ddf_colset ignoredcols,basiscols;
  ddf_rowrange i,k,m;
  ddf_rowindex newpos1;
  ddf_boolean success=ddf_FALSE;
  
  linrows=ddf_ImplicitLinearityRows(*M, error);
  if (*error!=ddf_NoError) goto _L99;
  
  m=(*M)->rowsize;
      
  set_uni((*M)->linset, (*M)->linset, linrows); 
      /* add the implicit linrows to the explicit linearity rows */

  /* To remove redundancy of the linearity part, 
     we need to compute the rank and a basis of the linearity part. */
  set_initialize(&ignoredrows,  (*M)->rowsize);
  set_initialize(&ignoredcols,  (*M)->colsize);
  set_compl(ignoredrows,  (*M)->linset);
  rank=ddf_MatrixRank(*M,ignoredrows,ignoredcols,&basisrows,&basiscols);
  set_diff(ignoredrows,  (*M)->linset, basisrows);
  ddf_MatrixRowsRemove2(M,ignoredrows,newpos);
  
  ddf_MatrixShiftupLinearity(M,&newpos1); 
 
  for (i=1; i<=m; i++){
    k=(*newpos)[i];
    if (k>0) {
      (*newpos)[i]=newpos1[k];
    }
  }
  
  *impl_linset=linrows;
  success=ddf_TRUE;
  free(newpos1);
  set_free(basisrows);
  set_free(basiscols);
  set_free(ignoredrows);
  set_free(ignoredcols);
_L99:
  return success;
}

ddf_boolean ddf_MatrixCanonicalize(ddf_MatrixPtr *M, ddf_rowset *impl_linset, ddf_rowset *redset, 
ddf_rowindex *newpos, ddf_ErrorType *error) /* 094 */
{
/* This is to find a canonical representation of a matrix *M by 
   recognizing all implicit linearities and all redundancies.  
   All implicit linearities will be returned by *impl_linset and
   redundancies will be returned by *redset.
*/
  ddf_rowrange i,k,m;
  ddf_rowindex newpos1,revpos;
  ddf_rowset redset1;
  ddf_boolean success=ddf_TRUE;
  
  m=(*M)->rowsize;
  set_initialize(redset, m);
  revpos=(long *)calloc(m+1,sizeof(long));
  
  success=ddf_MatrixCanonicalizeLinearity(M, impl_linset, newpos, error);

  if (!success) goto _L99;  
  
  for (i=1; i<=m; i++){
    k=(*newpos)[i];
    if (k>0) revpos[k]=i;  /* inverse of *newpos[] */
  }
 
  success=ddf_MatrixRedundancyRemove(M, &redset1, &newpos1, error);  /* 094 */
  
  if (!success) goto _L99;

  for (i=1; i<=m; i++){
    k=(*newpos)[i];
    if (k>0) {
      (*newpos)[i]=newpos1[k];
      if (newpos1[k]<0) (*newpos)[i]=-revpos[-newpos1[k]];  /* update the certificate of its duplicate removal. */
      if (set_member(k,redset1)) set_addelem(*redset, i);
    }
  }

_L99:
  set_free(redset1);
  free(newpos1);
  free(revpos);
  return success;
}


ddf_boolean ddf_ExistsRestrictedFace(ddf_MatrixPtr M, ddf_rowset R, ddf_rowset S, ddf_ErrorType *err)
/* 0.94 */
{
/* This function checkes if there is a point that satifies all the constraints of
the matrix M (interpreted as an H-representation) with additional equality contraints
specified by R and additional strict inequality constraints specified by S.
The set S is supposed to be disjoint from both R and M->linset.   When it is not,
the set S will be considered as S\(R U M->linset).
*/
  ddf_boolean answer=ddf_FALSE;
  ddf_LPPtr lp=NULL;

/*
  printf("\n--- ERF ---\n");
  printf("R = "); set_write(R); 
  printf("S = "); set_write(S);
*/
  
  lp=ddf_Matrix2Feasibility2(M, R, S, err);

  if (*err!=ddf_NoError) goto _L99;
 
/* Solve the LP by cdd LP solver. */
  ddf_LPSolve(lp, ddf_DualSimplex, err);  /* Solve the LP */
  if (*err!=ddf_NoError) goto _L99;
  if (lp->LPS==ddf_Optimal && ddf_Positive(lp->optvalue)) {
    answer=ddf_TRUE;
  } 

  ddf_FreeLPData(lp);
_L99:
  return answer;
}

ddf_boolean ddf_ExistsRestrictedFace2(ddf_MatrixPtr M, ddf_rowset R, ddf_rowset S, ddf_LPSolutionPtr *lps, ddf_ErrorType *err)
/* 0.94 */
{
/* This function checkes if there is a point that satifies all the constraints of
the matrix M (interpreted as an H-representation) with additional equality contraints
specified by R and additional strict inequality constraints specified by S.
The set S is supposed to be disjoint from both R and M->linset.   When it is not,
the set S will be considered as S\(R U M->linset).

This function returns a certificate of the answer in terms of the associated LP solutions.
*/
  ddf_boolean answer=ddf_FALSE;
  ddf_LPPtr lp=NULL;

/*
  printf("\n--- ERF ---\n");
  printf("R = "); set_write(R); 
  printf("S = "); set_write(S);
*/
  
  lp=ddf_Matrix2Feasibility2(M, R, S, err);

  if (*err!=ddf_NoError) goto _L99;
 
/* Solve the LP by cdd LP solver. */
  ddf_LPSolve(lp, ddf_DualSimplex, err);  /* Solve the LP */
  if (*err!=ddf_NoError) goto _L99;
  if (lp->LPS==ddf_Optimal && ddf_Positive(lp->optvalue)) {
    answer=ddf_TRUE;
  } 


  (*lps)=ddf_CopyLPSolution(lp);
  ddf_FreeLPData(lp);
_L99:
  return answer;
}

ddf_boolean ddf_FindRelativeInterior(ddf_MatrixPtr M, ddf_rowset *ImL, ddf_rowset *Lbasis, ddf_LPSolutionPtr *lps, ddf_ErrorType *err) 
/* 0.94 */
{
/* This function computes a point in the relative interior of the H-polyhedron given by M.
Even the representation is V-representation, it simply interprete M as H-representation.
lps returns the result of solving an LP whose solution is a relative interior point.
ImL returns all row indices of M that are implicit linearities, i.e. their inqualities
are satisfied by equality by all points in the polyhedron.  Lbasis returns a row basis
of the submatrix of M consisting of all linearities and implicit linearities.  This means
that the dimension of the polyhedron is M->colsize - set_card(Lbasis) -1.
*/

  ddf_rowset S;
  ddf_colset T, Lbasiscols;
  ddf_boolean success=ddf_FALSE;
  ddf_rowrange i;
  ddf_colrange rank;
  

  *ImL=ddf_ImplicitLinearityRows(M, err);

  if (*err!=ddf_NoError) goto _L99;

  set_initialize(&S, M->rowsize);   /* the empty set */
  for (i=1; i <=M->rowsize; i++) {
	if (!set_member(i, M->linset) && !set_member(i, *ImL)){
	  set_addelem(S, i);  /* all nonlinearity rows go to S  */
	}
  }
  if (ddf_ExistsRestrictedFace2(M, *ImL, S, lps, err)){
    /* printf("a relative interior point found\n"); */
    success=ddf_TRUE;
  }
  
  set_initialize(&T,  M->colsize); /* empty set */
  rank=ddf_MatrixRank(M,S,T,Lbasis,&Lbasiscols); /* the rank of the linearity submatrix of M.  */

  set_free(S);
  set_free(T);
  set_free(Lbasiscols);
  
_L99:
  return success;
}


ddf_rowrange ddf_RayShooting(ddf_MatrixPtr M, ddf_Arow p, ddf_Arow r)
{
/* 092, find the first inequality "hit" by a ray from an intpt.  */
  ddf_rowrange imin=-1,i,m;
  ddf_colrange j, d;
  ddf_Arow vecmin, vec;
  myfloat min,t1,t2,alpha, t1min;  
  ddf_boolean started=ddf_FALSE;
  ddf_boolean localdebug=ddf_FALSE;

  m=M->rowsize;
  d=M->colsize;
  if (!ddf_Equal(ddf_one, p[0])){
    fprintf(stderr, "Warning: RayShooting is called with a point with first coordinate not 1.\n");
    ddf_set(p[0],ddf_one);
  }
  if (!ddf_EqualToZero(r[0])){
    fprintf(stderr, "Warning: RayShooting is called with a direction with first coordinate not 0.\n");
    ddf_set(r[0],ddf_purezero);
  }

  ddf_init(alpha); ddf_init(min); ddf_init(t1); ddf_init(t2); ddf_init(t1min);
  ddf_InitializeArow(d,&vecmin);
  ddf_InitializeArow(d,&vec);

  for (i=1; i<=m; i++){
    ddf_InnerProduct(t1, d, M->matrix[i-1], p);
    if (ddf_Positive(t1)) {
      ddf_InnerProduct(t2, d, M->matrix[i-1], r);
      ddf_div(alpha, t2, t1);
      if (!started){
        imin=i;  ddf_set(min, alpha);
        ddf_set(t1min, t1);  /* store the denominator. */
        started=ddf_TRUE;
        if (localdebug) {
          fprintf(stderr," Level 1: imin = %ld and min = ", imin);
          ddf_WriteNumber(stderr, min);
          fprintf(stderr,"\n");
        }
      } else {
        if (ddf_Smaller(alpha, min)){
          imin=i;  ddf_set(min, alpha);
          ddf_set(t1min, t1);  /* store the denominator. */
          if (localdebug) {
            fprintf(stderr," Level 2: imin = %ld and min = ", imin);
            ddf_WriteNumber(stderr, min);
            fprintf(stderr,"\n");
          }
        } else {
          if (ddf_Equal(alpha, min)) { /* tie break */
            for (j=1; j<= d; j++){
              ddf_div(vecmin[j-1], M->matrix[imin-1][j-1], t1min);
              ddf_div(vec[j-1], M->matrix[i-1][j-1], t1);
            }
            if (ddf_LexSmaller(vec,vecmin, d)){
              imin=i;  ddf_set(min, alpha);
              ddf_set(t1min, t1);  /* store the denominator. */
              if (localdebug) {
                fprintf(stderr," Level 3: imin = %ld and min = ", imin);
                ddf_WriteNumber(stderr, min);
                fprintf(stderr,"\n");
              }
            }
          }
        }
      }       
    }
  }

  ddf_clear(alpha); ddf_clear(min); ddf_clear(t1); ddf_clear(t2); ddf_clear(t1min);
  ddf_FreeArow(d, vecmin);
  ddf_FreeArow(d, vec);
  return imin;
}

#ifdef ddf_GMPRATIONAL
void ddf_BasisStatusMaximize(ddf_rowrange m_size,ddf_colrange d_size,
    ddf_Amatrix A,ddf_Bmatrix T,ddf_rowset equalityset,
    ddf_rowrange objrow,ddf_colrange rhscol,ddf_LPStatusType LPS,
    myfloat *optvalue,ddf_Arow sol,ddf_Arow dsol,ddf_rowset posset, ddf_colindex nbindex,
    ddf_rowrange re,ddf_colrange se, ddf_colrange *nse, long *pivots, int *found, int *LPScorrect)
/*  This is just to check whether the status LPS of the basis given by 
nbindex with extra certificates se or re is correct.  It is done
by recomputing the basis inverse matrix T.  It does not solve the LP
when the status *LPS is undecided.  Thus the input is
m_size, d_size, A, equalityset, LPS, nbindex, re and se.
Other values will be recomputed from scratch.

The main purpose of the function is to verify the correctness
of the result of floating point computation with the GMP rational
arithmetics.
*/
{
  long pivots0,pivots1,fbasisrank;
  ddf_rowrange i,is;
  ddf_colrange s,senew,j;
  static ddf_rowindex bflag;
  static long mlast=0;
  static ddf_rowindex OrderVector;  /* the permutation vector to store a preordered row indices */
  unsigned int rseed=1;
  myfloat val;
  ddf_colindex nbtemp;
  ddf_LPStatusType ddlps;
  ddf_boolean localdebug=ddf_FALSE;

  if (ddf_debug) localdebug=ddf_debug;
  if (localdebug){
     printf("\nEvaluating ddf_BasisStatusMaximize:\n");
  }
  ddf_init(val);
  nbtemp=(long *) calloc(d_size+1,sizeof(long));
  for (i=0; i<= 4; i++) pivots[i]=0;
  if (bflag==NULL || mlast!=m_size){
     if (mlast!=m_size && mlast>0) {
       free(bflag);   /* called previously with different m_size */
       free(OrderVector);
     }
     bflag=(long *) calloc(m_size+1,sizeof(long));
     OrderVector=(long *)calloc(m_size+1,sizeof(long)); 
     /* initialize only for the first time or when a larger space is needed */
      mlast=m_size;
  }

  /* Initializing control variables. */
  ddf_ComputeRowOrderVector2(m_size,d_size,A,OrderVector,ddf_MinIndex,rseed);

  pivots1=0;

  ddf_ResetTableau(m_size,d_size,T,nbtemp,bflag,objrow,rhscol);

  if (localdebug){
     printf("\nnbindex:");
     for (j=1; j<=d_size; j++) printf(" %ld", nbindex[j]);
     printf("\n");
     printf("re = %ld,   se=%ld\n", re, se);
  }
  
  is=nbindex[se];
  if (localdebug) printf("se=%ld,  is=%ld\n", se, is);
  
  fbasisrank=d_size-1;
  for (j=1; j<=d_size; j++){
    if (nbindex[j]<0) fbasisrank=fbasisrank-1;
	/* fbasisrank=the basis rank computed by floating-point */
  }

  if (fbasisrank<d_size-1) {
    if (localdebug) {
	  printf("d_size = %ld, the size of basis = %ld\n", d_size, fbasisrank);
	  printf("ddf_BasisStatusMaximize: the size of basis is smaller than d-1.\nIt is safer to run the LP solver with GMP\n");
	}
	*found=ddf_FALSE;
	goto _L99;
     /* Suspicious case.  Rerun the LP solver with GMP. */
  }



  ddf_FindLPBasis2(m_size,d_size,A,T,OrderVector, equalityset,nbindex,bflag,
      objrow,rhscol,&s,found,&pivots0);

/* set up the new se column and corresponding variable */
  senew=bflag[is];
  is=nbindex[senew];
  if (localdebug) printf("new se=%ld,  is=%ld\n", senew, is);
      
  pivots[4]=pivots0;  /*GMP postopt pivots */
  ddf_statBSpivots+=pivots0;

  if (!(*found)){
    if (localdebug) {
       printf("ddf_BasisStatusMaximize: a specified basis DOES NOT exist.\n");
    }

       goto _L99;
     /* No speficied LP basis is found. */
  }

  if (localdebug) {
    printf("ddf_BasisStatusMaximize: a specified basis exists.\n");
    if (m_size <=100 && d_size <=30)
    ddf_WriteTableau(stdout,m_size,d_size,A,T,nbindex,bflag);
  }

  /* Check whether a recomputed basis is of the type specified by LPS */
  *LPScorrect=ddf_TRUE;
  switch (LPS){
     case ddf_Optimal: 
       for (i=1; i<=m_size; i++) {
         if (i!=objrow && bflag[i]==-1) {  /* i is a basic variable */
            ddf_TableauEntry(&val,m_size,d_size,A,T,i,rhscol);
            if (ddf_Negative(val)) {
               if (localdebug) printf("RHS entry for %ld is negative\n", i);
               *LPScorrect=ddf_FALSE;
               break;
            }
          } else if (bflag[i] >0) { /* i is nonbasic variable */
            ddf_TableauEntry(&val,m_size,d_size,A,T,objrow,bflag[i]);
            if (ddf_Positive(val)) {
               if (localdebug) printf("Reduced cost entry for %ld is positive\n", i);
               *LPScorrect=ddf_FALSE;
               break;
            }
          }
       };
       break;
     case ddf_Inconsistent: 
       for (j=1; j<=d_size; j++){
          ddf_TableauEntry(&val,m_size,d_size,A,T,re,j);
          if (j==rhscol){
             if (ddf_Nonnegative(val)){
               if (localdebug) printf("RHS entry for %ld is nonnegative\n", re);
               *LPScorrect=ddf_FALSE;
               break;             
             }
           } else if (ddf_Positive(val)){
               if (localdebug) printf("the row entry for(%ld, %ld) is positive\n", re, j);
               *LPScorrect=ddf_FALSE;
               break;             
           }
       };
       break;
     case ddf_DualInconsistent:
        for (i=1; i<=m_size; i++){
          ddf_TableauEntry(&val,m_size,d_size,A,T,i,bflag[is]);
          if (i==objrow){
             if (ddf_Nonpositive(val)){
               if (localdebug) printf("Reduced cost entry for %ld is nonpositive\n", bflag[is]);
               *LPScorrect=ddf_FALSE;
               break;             
             }
           } else if (ddf_Negative(val)){
               if (localdebug) printf("the column entry for(%ld, %ld) is positive\n", i, bflag[is]);
               *LPScorrect=ddf_FALSE;
               break;             
           }
       };
       break;
;
     default: break;
  }

  ddlps=LPSf2LPS(LPS);

  ddf_SetSolutions(m_size,d_size,A,T,
   objrow,rhscol,ddlps,optvalue,sol,dsol,posset,nbindex,re,senew,bflag);
  *nse=senew;

  
_L99:
  ddf_clear(val);
  free(nbtemp);
}

void ddf_BasisStatusMinimize(ddf_rowrange m_size,ddf_colrange d_size,
    ddf_Amatrix A,ddf_Bmatrix T,ddf_rowset equalityset,
    ddf_rowrange objrow,ddf_colrange rhscol,ddf_LPStatusType LPS,
    myfloat *optvalue,ddf_Arow sol,ddf_Arow dsol, ddf_rowset posset, ddf_colindex nbindex,
    ddf_rowrange re,ddf_colrange se,ddf_colrange *nse,long *pivots, int *found, int *LPScorrect)
{
   ddf_colrange j;
   
   for (j=1; j<=d_size; j++) ddf_neg(A[objrow-1][j-1],A[objrow-1][j-1]);
   ddf_BasisStatusMaximize(m_size,d_size,A,T,equalityset, objrow,rhscol,
     LPS,optvalue,sol,dsol,posset,nbindex,re,se,nse,pivots,found,LPScorrect);
   ddf_neg(*optvalue,*optvalue);
   for (j=1; j<=d_size; j++){
	if (LPS!=ddf_Inconsistent) {
	   /* Inconsistent certificate stays valid for minimization, 0.94e */
       ddf_neg(dsol[j-1],dsol[j-1]);
	 }
     ddf_neg(A[objrow-1][j-1],A[objrow-1][j-1]);
   }
}
#endif

/* end of cddlp.c */

