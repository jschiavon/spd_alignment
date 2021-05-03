import numpy as np

# Functions
_isqrt = lambda x: 1. / np.sqrt(x)
_funs = {'sqrt': np.sqrt,
        'isqrt': _isqrt,
        'log': np.log,
        'exp': np.exp}

def _transform_mat(X, func='sqrt'):
    """
    Applies a transformation to a SPD matrix by means of the eigenvalues.

    This function compute the eigenvalue decomposition of a SPD matrix and
    returns the matrix obtained by applying the transformation func to the 
    eigenvalues before reconstructing the matrix, i.e. returns
    `V * func(Lambda) * V'` where `V * Lambda * V'` is the eigenvalue decomposition
    of `X`
    """
    u, v = np.linalg.eigh(X)
    return np.einsum('...ij,...j,...kj', v, _funs[func](u), v)


def norm_frob_squared(X):
    """
    Computes the squared Frobenius norm of a matrix.
    """
    return np.einsum('...ji,...ji', X, X)


def dist_frob_squared(X, Y):
    """
    Computes the squared distance, induced by frobenius norm, between two matrices.
    """
    return norm_frob_squared(X - Y)


def dist_riem_squared(X, Y):
    """
    Computes the squared distance, induced by riemmanian norm, between two SPD matrices.
    """
    x = _transform_mat(X, 'isqrt')
    mid = np.einsum('...ij,...jk,...kl', x, Y, x)
    return norm_frob_squared(_transform_mat(mid, 'log'))


def norm_riem_squared(X):
    """
    Computes the squared Riemannian norm of a SPD matrix.
    """
    x = _transform_mat(X, 'log')
    return norm_frob_squared(x)


def rotate(X, Omega):
    """
    Rotates a SPD matrix `X` with an orthogonal matrix `Omega`.
    """
    return np.einsum('...ij,...jk,...lk', Omega, X, Omega)



def rWc(kap, m, rng):
    b = -2 * kap + np.sqrt(4 * kap**2 + (m - 1)**2) / (m - 1)
    x0 = (1 - b) / (1 + b)
    c = kap * x0 + (m - 1) * np.log(1 - x0**2)
    while True:
        Z = rng.beta((m - 1) / 2, (m - 1) / 2)
        W = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
        U = rng.random()
        if (kap * W + (m - 1) * np.log(1 - x0*W) - c) > np.log(U):
            break
    return W



#' @export NullC
# NullC <-
# function(M)
# {
#   #modified from package "MASS" 
#   #MASS version : Null(matrix(0,4,2))  returns a 4*2 matrix
#   #this version : NullC(matrix(0,4,2)) returns diag(4)

#   tmp <- qr(M)
#   set <- if (tmp$rank == 0L)
#       1L:nrow(M)
#   else -(1L:tmp$rank)
#   qr.Q(tmp, complete = TRUE)[, set, drop = FALSE]
# }

def nullC(M):
    tmp_q, tmp_r = np.linalg.qr(M)



def sample_vMFvector(kmu, rng):
    kap = np.linalg.norm(kmu)
    mu = kmu / kap
    m = kmu.shape[-1]

    if kap == 0:
        u = rng.normal(size=(m))
        return u / np.linalg.norm(u)
    
    if m == 1:
        return -1 ** (rng.binomial(1, 1 / (1 + np.exp(2 * kap * mu))))
    
    W = rWc(kap, m, rng)
    V = rng.normal(m - 1)
    V = V / np.linalg.norm(V)
    x = np.append(np.sqrt(1 - W**2) * V, W)
    u = 1





# rmf.vector <-
# function(kmu)
# {
#   #simulate from the vector mf distribution as described in Wood(1994)
#   kap<-sqrt(sum(kmu^2)) ; mu<-kmu/kap ; m<-length(mu)
#   if(kap==0){ u<-rnorm(length(kmu)) ; u<-matrix(u/sqrt(sum(u^2)),m,1) }
#   if(kap>0)
#   {
#     if(m==1){ u<- (-1)^rbinom( 1,1,1/(1+exp(2*kap*mu))) }
#     if(m>1)
#     {
#       W<-rW(kap,m)
#       V<-rnorm(m-1) ;  V<-V/sqrt(sum(V^2))
#       x<-c((1-W^2)^.5*t(V),W)
#       u<-cbind( NullC(mu),mu)%*%x
#     }
#   }
#   u
# }




def sample_vMF(M, rng):
    u, l, v = np.linalg.svd(M)

    H = np.einsum('ij,j->ij', u, l)
    m, R = H.shape

    cmet = False
    rej = 0
    while ~cmet:
        U = np.zeros(shape=(m, R))
        U[:, 0] = 1


# rmf.matrix <-
# function(M)
# {
#   if(dim(M)[2]==1) { X<-rmf.vector(M) } 
#   if(dim(M)[2]>1) 
#   {
#     #simulate from the matrix mf distribution using the rejection 
#     #sampler as described in Hoff(2009)
#     svdM<-svd(M)
#     H<-svdM$u%*%diag(svdM$d)
#     m<-dim(H)[1] ; R<-dim(H)[2]

#     cmet<-FALSE
#     rej<-0
#     while(!cmet)
#     {
#       U<-matrix(0,m,R)
#       U[,1]<-rmf.vector(H[,1])
#       lr<-0

#       for(j in seq(2,R,length=R-1))
#       {
#         N<-NullC(U[,seq(1,j-1,length=j-1)])
#         x<-rmf.vector(t(N)%*%H[,j])
#         U[,j]<- N%*%x

#         if(svdM$d[j]>0) 
#         {
#           xn<- sqrt(sum( (t(N)%*%H[,j])^2))
#           xd<- sqrt(sum( H[,j]^2 ))
#           lbr<-  log(besselI(xn, .5*(m-j-1),expon.scaled=TRUE))-
#                  log(besselI(xd, .5*(m-j-1),expon.scaled=TRUE))
#           if(is.na(lbr)){lbr<- .5*(log(xd) - log(xn)) }
#           lr<- lr+ lbr + (xn-xd) + .5*(m-j-1)*( log(xd)-log(xn) )
#         }
#       }

#       cmet<- (log(runif(1)) <  lr ) ; rej<-rej+(1-1*cmet)
#     }
#     X<-U%*%t(svd(M)$v)
#   }
#   X
# }