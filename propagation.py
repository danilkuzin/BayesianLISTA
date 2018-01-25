class Gaussian:
    def __init__(self, m=0, v=1):
        self.m = m
        self.v = v

class Gaussian_deriv:
    def __init__(self, m_xm=0, m_ym=0, v_xv=0, v_yv=0):
        self.m_xm = m_xm
        self.m_ym = m_ym
        self.v_xv = v_xv
        self.v_yv = v_yv
        

def sum_gaussians(x, y):
    res = Gaussian()
    res.m = x.m + y.m
    res.v = x.v + y.v
    
    grad = Gaussian_deriv()
    grad.m_xm = 1
    grad.m_ym = 1
    grad.v_xv = 1
    grad.v_yv = 1
    
    return res, grad

# this is wrong, product of Gaussians is not a Gaussian
def prod_gaussians(x, y):
    res = Gaussian()
    res.m = (y.v * x.m + x.v * y.m) / (x.v + y.v)
    res.v = (x.v * y.v) / (x.v + y.v)
    
    return res

def prod_gaussian_const(x, alpha):
    res = Gaussian()
    res.m = x.m * alpha
    res.v = x.v * (alpha**2)
    
    return res

def prod_gaussian_matrix_const_vector(W, beta):
    D, K = W.m.shape
            
    return res

def linear_gaussian_matrix(b, S, beta):
    
    return res