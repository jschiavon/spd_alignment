import jax.numpy as jnp
from jax import jit, partial, random, vmap
from jax.ops import index, index_update
from jax.lax import fori_loop



@jit
def sqrtm(x):
    u, v = jnp.linalg.eigh(x)
    return jnp.einsum('...ij,...j,...kj', v, jnp.sqrt(u), v)

@jit
def isqrtm(x):
    u, v = jnp.linalg.eigh(x)
    return jnp.einsum('...ij,...j,...kj', v, 1/jnp.sqrt(u), v)

@jit
def logm(x):
    u, v = jnp.linalg.eigh(x)
    return jnp.einsum('...ij,...j,...kj', v, jnp.log(u), v)

@jit
def expm(x):
    u, v = jnp.linalg.eigh(x)
    return jnp.einsum('...ij,...j,...kj', v, jnp.exp(u), v)

@jit
def dist(x, y):
    ix = isqrtm(x)
    mid = jnp.einsum('...ij,...jk,...kl', ix, y, ix)
    return jnp.linalg.norm(logm(mid), axis=(-2, -1))

@jit
def dist_sq(x, y):
    ix = isqrtm(x)
    mid = jnp.einsum('...ij,...jk,...kl', ix, y, ix)
    logmid = logm(mid)
    return jnp.einsum('...ji,...ji', logmid, logmid)

@jit
def cost(x, sigmas):
    return jnp.sqrt(jnp.sum(dist(x, sigmas) ** 2))


@partial(jit, static_argnums=(1))
def optimal_reference(X: jnp.array, return_eigs: bool):
    u, vs = jnp.linalg.eigh(X)
    eigval = jnp.exp(jnp.sum(jnp.log(u), axis=0) / X.shape[0])
    U, _, V = jnp.linalg.svd(jnp.sum(vs, axis=0))
    eigvec = jnp.einsum('ij,jk', U, V)
    ref = jnp.einsum('ij,j,kj', eigvec, eigval, eigvec)
    if return_eigs:
        return ref, eigval, eigvec
    return ref

@jit
def optimal_rotation(X, M):
    _, g_m = jnp.linalg.eigh(M)
    _, g_x = jnp.linalg.eigh(X)
    Omega = jnp.einsum('...ij,...kj', g_m, g_x) 
    return jnp.einsum('...ij,...jk,...lk', Omega, X, Omega)


@partial(jit, static_argnums=(2,3))
def make_distanceMatrix(points, idx, distance, n):
    distmatrix = jnp.zeros(shape=(n, n))
    
    def bodyfun(i, dists):
        j, k = idx[i]
        return dists.at[j, k].set(distance(points[j], points[k]))
    
    distmatrix = fori_loop(0, len(idx), bodyfun, distmatrix)
    return distmatrix

@jit
def renormalize_kernel(kernel, alpha):    
    q = jnp.power(jnp.dot(kernel, jnp.ones(shape=(len(kernel)))), alpha)
    K = jnp.divide(kernel, jnp.outer(q,q))    
    return K        

@jit
def make_kernelMatrix(distmatrix, eps):
    kernel = jnp.exp(-distmatrix**2/eps)            
    return kernel

@jit
def make_transitionMatrix(kernel):    
    d = jnp.sqrt(jnp.dot(kernel, jnp.ones(shape=(len(kernel)))))
    P = jnp.divide(kernel, jnp.outer(d, d))    
    return P


def get_diffusionEmbedding(points=[], distance=[], distmatrix=None, alpha=1.0, tdiff=0, eps=None):
    n = len(points)
    if distmatrix is None:
        idx = jnp.array([[i, j] for i in range(n) for j in range(n)])
        d = make_distanceMatrix(points=points, idx=idx, distance=distance, n=n)
    else:
        d = distmatrix

    if eps is None:
        # using heuristic from the R package for diffusion maps
        eps = 2*jnp.median(d)**2      
    
    K = make_kernelMatrix(distmatrix=d, eps=eps)
    Kr = renormalize_kernel(K, alpha=alpha)            
    P = make_transitionMatrix(Kr)
    u,s,v = jnp.linalg.svd(P)    
    
    phi = u
    for i in range(len(u)):
        phi.at[:,i].set((s[i]**tdiff)*jnp.divide(u[:,i], u[:,0]))
        
    return phi, s


@jit
def is_sorted(a):
    return jnp.all(jnp.diff(a) >= 0)


@partial(jit, static_argnums=(2,3))
def proposal(key, mean, sig, p):
    return random.multivariate_normal(key, mean=mean, cov= 2 * sig * jnp.identity(p))


@partial(jit, static_argnums=(1,2))
def logf(r, sig, p):
    first = - jnp.sum(r**2) / (2 * sig**2)
    second = jnp.sum(jnp.log(jnp.sinh(jnp.abs(r[:, None] - r[None, :]))[jnp.tril_indices(p, k=-1)]))
    return first + second


@partial(jit, static_argnums=(2,3))
def density_ratio(rprime, r, sig, p):
    return logf(rprime, sig, p) - logf(r, sig, p)


@partial(jit, static_argnums=(2,3))
def newr(key, r, sig, p):
    key1, key2 = random.split(key)
    
    rprime = proposal(key1, r, sig, p)
    alpha = density_ratio(rprime, r, sig, p)
    u = random.uniform(key2)
    
    return jnp.where(u < jnp.exp(alpha),
                     rprime,
                     r)

@partial(jit, static_argnums=(1,2,3,4))
def generate_rs(key, sample_size, sig, p, sort=False):
    key1, key2 = random.split(key)
    r0 = proposal(key1, jnp.zeros(shape=(p)), sig, p)
    r0 = jnp.where(sort, r0.sort(), r0)
    
    r = jnp.zeros(shape=(sample_size, p))
    r = index_update(r, 0, r0)
    ks = random.split(key2, sample_size)
    
    def bodyfun(i, rvals):
        rprime = newr(ks[i], rvals[i], sig, p)
        rprime = jnp.where(sort, rprime.sort(), rprime)
        return index_update(rvals, i+1, rprime)

    r = fori_loop(0, sample_size, bodyfun, r)

    return r


def Wishart(key, dof, scale, shape=None):
    if scale is None:
        scale = jnp.eye(shape)
    batch_shape = ()
    if jnp.ndim(scale) > 2:
        batch_shape = scale.shape[:-2]
    p = scale.shape[-1]
    
    if dof is None:
        dof = p
    if jnp.ndim(dof) > 0:
        raise ValueError("only scalar dof implemented")
    if ~(int(dof) == dof):
        raise ValueError("dof should be integer-like (i.e. int(dof) == dof should return true)")
    else:
        dof = int(dof)
    
    if shape is not None:
        if batch_shape != ():
            assert batch_shape == shape, "Disagreement in batch shape between scale and shape"
        else:
            batch_shape = shape
    
    mn = jnp.zeros(shape=batch_shape + (p, ))
    mvn_shape = (dof,) + batch_shape
    
    mvn = random.multivariate_normal(key, mean=mn, cov=scale, shape=mvn_shape)
    if jnp.ndim(mvn) > 2:
        mvn = jnp.swapaxes(mvn, 0, -2)
    
    S = jnp.einsum('...ji,...jk', mvn, mvn)
    
    return S


def InvWishart(key, dof, scale, shape=None):
    W = Wishart(key, dof, scale, shape)
    return jnp.linalg.inv(W)


def perturbate(key, original_sample, dof):
    if dof is None:
        return original_sample
    new_sample = Wishart(key, dof, original_sample / dof)
    return new_sample


def generate_sample(key, center, p, sample_size, sig, dof=None, diag=False, sort=False, thinning=20, burnin=2000, chains=1, verbose=False):
    total_size = sample_size * thinning + burnin
    key = random.split(key, 3)
    
    gen_rs = jit(lambda k: generate_rs(k, total_size, sig=sig, p=p, sort=sort))
    
    # SAMPLE EIGENVALUES
    if verbose: print("Sampling eigenvalues....")
    if chains == 1:
        r = gen_rs(key[0])
        r = r[burnin::thinning]
    else:
        k, *keys = random.split(key[0], chains + 1)
        r = vmap(gen_rs)(jnp.array(keys))
        r = r[:, burnin::thinning].reshape(-1, p)
        r = r[random.choice(k, len(r), shape=(sample_size,))]
    lmbd = jnp.exp(r)

    if diag:
        mats = jnp.array([jnp.diag(l) for l in lmbd])
    else:
        # SAMPLE EIGENVECTORS
        if verbose: print("Sampling eigenvectors....")
        O = random.normal(key[1], shape=(sample_size, p, p))
        U = jnp.linalg.qr(O)[0]

        # CENTERED MATS
        if verbose: print("Computing matrices....")
        mats = jnp.einsum('...ij,...j,...kj', U, lmbd, U)
    
    # GENERATE UNCENTERED MATS
    mats = jnp.einsum('ij,...jk,kl->...il', sqrtm(center), mats, sqrtm(center))

    # ADD WISHART PERTURBATION
    if dof is not None:
        if verbose: print("Perturbing matrices....")
        mats = perturbate(key[2], mats, dof)

    return mats


def generate_center(key, p, maxexp=5, minexp=None, diag=False):
    if minexp is None:
        minexp = - maxexp
    key1, key2 = random.split(key)
    center_diag = jnp.exp(random.uniform(key1, shape=(p,), minval=minexp, maxval=maxexp).sort())
    if diag:
        return jnp.diag(center_diag)
    U = jnp.linalg.qr(random.normal(key2, shape=(p, p)))[0]
    return jnp.einsum('...ij,...j,...kj', U, center_diag, U)


