import numpy as np
import scipy.special as scsp
from pycop.bivariate.copula import copula
# import jax.numpy as jnp
# import jax.scipy.special as jscsp
# from jax import grad
# from jax import config
# config.update("jax_enable_x64", True)

# Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

# Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

# Frank Copula
def frank_copula(u, v, theta):
    num = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
    denom = np.exp(-theta) - 1
    return -np.log(1 + num / denom) / theta

# Joe Copula
def joe_copula(u, v, theta):
    x = (1 - u)**theta
    y = (1 - v)**theta
    return 1 - (x + y - x*y)**(1/theta)

# Galambos Copula
def galambos_copula(u, v, theta):
    return u*v*np.exp(-(((-np.log(u))**(-theta) + (-np.log(v))**(-theta))**(-1/theta)))

# FGM Copula
def fgm_copula(u, v, theta):
    return u * v * (1 + theta * (1 - u) * (1 - v))

# Plackett Copula
def plackett_copula(u, v, theta):
    eta = theta - 1

    if np.isclose(theta, 1.0):  # use Taylor-expansion for theta close to one
        return u*v + eta*u*v*(1 - u)*(1 - v)

    eta = theta - 1
    upv = u + v
    num = 1 + eta * upv - np.sqrt((1 + eta * upv)**2 - 4 * u * v * theta * eta)
    denom = 2 * eta

    return num / denom

# BB1 Copula
def bb1_copula(u, v, theta, delta):
    return (1 + ((u**(-theta) - 1)**delta + (v**(-theta) - 1)**delta)**(1/delta))**(-1/theta)

# BB2 Copula
def bb2_copula(u, v, theta, delta):
    x = theta*(u**(-delta) - 1)
    y = theta*(v**(-delta) - 1)
    logt = scsp.logsumexp(np.array([x, y, 0.0]), 0, np.array([1.0, 1.0, -1.0]))
    return (1 + (1/theta)*logt)**(-1/delta)

class archimedean(copula):
    """
    # Creates an Archimedean copula objects
    Source for the CDF and PDF functions:
    Joe, H. (2014). Dependence modeling with copulas. CRC press.
    Chapter 4: Parametric copula families and properties (p.159)

    ...

    Attributes
    ----------
    family : str
        The name of the Archimedean copula function.
    type : str
        The type of copula = "archimedean".
    bounds_param : list
        A list that contains the domain of the parameter(s) in a tuple.
        Exemple : [(lower, upper)]
    parameters_start : array
        Value(s) of the initial guess when estimating the copula parameter(s).
        It represents the parameter `x0` in the `scipy.optimize.minimize` function.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the Cumulative Distribution Function (CDF).
    get_pdf(u, v, param)
        Computes the Probability Density Function (PDF).
    LTDC(theta)
        Computes the Lower Tail Dependence Coefficient (TDC).
    UTDC(theta)
        Computes the upper TDC.
    """

    Archimedean_families = [
        'clayton', 'gumbel', 'frank', 'joe', 'galambos','fgm', 'plackett',
        'rgumbel', 'rclayton', 'rjoe','rgalambos', 'BB1', 'BB2']


    def __init__(self, family):
        """
        Parameters
        ----------
        family : str
            The name of the Archimedean copula function.

        Raises
        ------
        ValueError
            If the given `family` is not supported.
        """

        # the `archimedean` copula class inherit the `copula` class
        super().__init__()
        self.family = family
        self.type = "archimedean"

        if family  in ['clayton', 'galambos', 'plackett', 'rclayton', 'rgalambos'] :
            self.bounds_param = [(1e-6, None)]
            self.parameters_start = np.array(0.5)

        elif family in ['gumbel', 'joe', 'rgumbel', 'rjoe'] :
            self.bounds_param = [(1.0, None)]
            self.parameters_start = np.array(1.5)

        elif family == 'frank':
            self.bounds_param = [(None, None)]
            self.parameters_start = np.array(2.0)

        elif family == 'fgm':
            self.bounds_param = [(-1.0, 1.0 - 1e-6)]
            self.parameters_start = np.array(0.0)

        elif family  in ['BB1'] :
            self.bounds_param = [(1e-6, None), (1.0, None)]
            self.parameters_start = (np.array(0.5), np.array(1.5))

        elif family  in ['BB2'] :
            self.bounds_param = [(1e-6, None), (1e-6, None)]
            self.parameters_start = (np.array(1.0), np.array(1.0))
        else:
            print("family \"%s\" not in list: %s" % (family, archimedean.Archimedean_families) )
            raise ValueError

    def get_grad_cdf(self, u, v, param):
        """
        # Computes the gradient of the CDF with respect to u and v

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            A list that contains the copula parameter(s) (float)
        """

        p0 = param[0]

        if self.family == 'clayton':
            t1 = (-1 + v**(-p0) + u**(-p0))
            return (1/(u*u**p0*t1*t1**(1/p0)),
                    1/(v*v**p0*t1*t1**(1/p0)))

        elif self.family == 'rclayton':
            return tuple(1 - x for x in archimedean(family='clayton').get_grad_cdf((1 - u),(1 - v), param))

        elif self.family == 'gumbel':
            logu, logv = np.log(u), np.log(v)
            mlogup, mlogvp = (-logu)**p0, (-logv)**p0
            mlupv = (mlogup + mlogvp)
            t1 = mlupv**(1/p0)*np.exp(-mlupv**(1/p0))
            return (-mlogup*t1/(u*mlupv*logu),
                    -mlogvp*t1/(v*mlupv*logv))

        elif self.family == 'rgumbel':
            return tuple(1 - x for x in archimedean(family='gumbel').get_grad_cdf((1 - u),(1 - v), param))

        elif self.family == 'frank':
            expu, expv, expp = np.exp(-p0*u), np.exp(-p0*v), np.exp(-p0)
            t1 = ((-1 + expp)*(1 + (-1 + expu)*(-1 + expv)/(-1 + expp)))
            return ((-1 + expv)*expu/t1,
                    (-1 + expu)*expv/t1)

        elif self.family == 'joe':
            u_ = (1 - u) ** p0
            v_ = (1 - v) ** p0
            t1 = (-u_*v_ + u_ + v_)**((1 - p0)/p0)
            return ((1 - u)**(p0 - 1)*(1 - v_)*t1,
                    (1 - v)**(p0 - 1)*(1 - u_)*t1)

        elif self.family == 'rjoe':
            return tuple(1 - x for x in archimedean(family='joe').get_grad_cdf((1 - u),(1 - v), param))

        elif self.family == 'galambos':
            logu, logv = np.log(u), np.log(v)
            mlogup, mlogvp = (-logu)**p0, (-logv)**p0
            mlupv = (mlogup + mlogvp)
            x_ = (mlupv/(mlogup*mlogvp))**(1/p0)
            t1 = np.exp((x_)**(-1/p0))

            return (v*(mlogvp/(x_*mlupv*logu) + 1)*t1,
                    u*(mlogup/(x_*mlupv*logv) + 1)*t1)

        elif self.family == 'rgalambos':
            return tuple(1 - x for x in archimedean(family='galambos').get_grad_cdf((1 - u),(1 - v), param))

        elif self.family == 'fgm':
            um1, vm1 = (u - 1), (v - 1)

            return (v*(p0*u*vm1 + p0*um1*vm1 + 1),
                    u*(p0*v*um1 + p0*um1*vm1 + 1))

        elif self.family == 'plackett':
            eta = p0 - 1
            upv = (u + v)
            t1 = (-4*p0*u*v*eta + (eta*upv + 1)**2)**0.5

            return ((1.0*p0*v - 0.5*eta*upv + 0.5*t1 - 0.5)/t1,
                    (1.0*p0*u - 0.5*eta*upv + 0.5*t1 - 0.5)/t1)

        elif self.family == 'BB1':
            p1 = param[1]

            U = -1 + u**(-p1)
            V = -1 + v**(-p1)
            Up = -1 + u**p1
            Vp = -1 + v**p1

            x = U**p0 + V**p0 + 1

            A = x**(1/(p0*p1))
            Bu = Up*u*x
            Bv = Vp*v*x

            cdf_u = -(1/A)*(1/Bu)*(U**p0)
            cdf_v = -(1/A)*(1/Bv)*(V**p0)

            return cdf_u, cdf_v


        elif self.family == 'BB2':
            p1 = param[1]

            x = p0/u**p1
            y = p0/v**p1

            exp_ymx = np.exp(y - x)
            exp_xmy = np.exp(x - y)
            exp_pmx = np.exp(p0 - x)
            exp_pmy = np.exp(p0 - y)
            logs = scsp.logsumexp([x, y, p0], 0, [1, 1, -1])

            A = logs*(logs/p0)**(1/p1)

            cdf_u = p0*u**(-p1 - 1)/(A + exp_ymx*A - A*exp_pmx)
            cdf_v = p0*v**(-p1 - 1)/(A + exp_xmy*A - A*exp_pmy)

            return cdf_u, cdf_v

    def get_cdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            A list that contains the copula parameter(s) (float)
        """

        if u == 0.0 or v == 0.0:
            return 0.0

        if u == 1.0 and v == 1.0:
            return 1.0

        theta = param[0]

        if self.family == 'clayton':
            return clayton_copula(u, v, theta)

        elif self.family == 'rclayton':
            return u + v - 1 + archimedean(family='clayton').get_cdf(1 - u, 1 - v, param)

        elif self.family == 'gumbel':
            return gumbel_copula(u, v, theta)

        elif self.family == 'rgumbel':
            return u + v - 1 + archimedean(family='gumbel').get_cdf(1 - u, 1 - v, param)

        elif self.family == 'frank':
            return frank_copula(u, v, theta)

        elif self.family == 'joe':
            return joe_copula(u, v, theta)

        elif self.family == 'rjoe':
            return u + v - 1 + archimedean(family='joe').get_cdf(1 - u, 1 - v, param)

        elif self.family == 'galambos':
            return galambos_copula(u, v, theta)

        elif self.family == 'rgalambos':
            return u + v - 1 + archimedean(family='galambos').get_cdf(1 - u, 1 - v, param)

        elif self.family == 'fgm':
            return fgm_copula(u, v, theta)

        elif self.family == 'plackett':
            return plackett_copula(u, v, theta)

        elif self.family == 'BB1':
            delta = param[1]
            return bb1_copula(u, v, theta, delta)

        elif self.family == 'BB2':
            delta = param[1]
            return bb2_copula(u, v, theta, delta)

    def get_pdf(self, u, v, param):
        """
        # Computes the PDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            A list that contains the copula parameter(s) (float)
        """

        if self.family == 'clayton':
            term1 = (param[0] + 1) * (u * v) ** (-param[0] - 1)
            term2 = (u ** (-param[0]) + v ** (-param[0]) - 1) ** (-2 - 1 / param[0])
            return term1 * term2

        if self.family == 'rclayton':
            return archimedean(family='clayton').get_pdf((1 - u),(1 - v), param)

        elif self.family == 'gumbel':
            term1 = np.power(np.multiply(u, v), -1)
            tmp = np.power(-np.log(u), param[0]) + np.power(-np.log(v), param[0])
            term2 = np.power(tmp, -2 + 2.0 / param[0])
            term3 = np.power(np.multiply(np.log(u), np.log(v)), param[0] - 1)
            term4 = 1 + (param[0] - 1) * np.power(tmp, -1 / param[0])
            return archimedean(family='gumbel').get_cdf(u,v, param) * term1 * term2 * term3 * term4

        if self.family == 'rgumbel':
            return archimedean(family='gumbel').get_pdf((1 - u), (1 - v), param)

        elif self.family == 'frank':
            term1 = param[0] * (1 - np.exp(-param[0])) * np.exp(-param[0] * (u + v))
            term2 = (1 - np.exp(-param[0]) - (1 - np.exp(-param[0] * u)) \
                    * (1 - np.exp(-param[0] * v))) ** 2
            return term1 / term2

        elif self.family == 'joe':
            u_ = (1 - u) ** param[0]
            v_ = (1 - v) ** param[0]
            term1 = (u_ + v_ - u_ * v_) ** (-2 + 1 / param[0])
            term2 = ((1 - u) ** (param[0] - 1)) * ((1 - v) ** (param[0] - 1))
            term3 = param[0] - 1 + u_ + v_ + u_ * v_
            return term1 * term2 * term3

        if self.family == 'rjoe':
            return archimedean(family='joe').get_pdf((1 - u),(1 - v), param)

        elif self.family == 'galambos':
            x = -np.log(u)
            y = -np.log(v)
            term1 = self.get_cdf(u, v, param) / (v * u)
            term2 = 1 - ((x ** (-param[0]) + y ** (-param[0])) ** (-1 - 1 / param[0])) \
                    * (x ** (-param[0] - 1) + y ** (-param[0] - 1))
            term3 = ((x ** (-param[0]) + y ** (-param[0])) ** (-2 - 1 / param[0])) \
                    * ((x * y) ** (-param[0] - 1))
            term4 = 1 + param[0] + ((x ** (-param[0]) + y ** (-param[0])) ** (-1 / param[0]))
            return term1 * term2 + term3 * term4

        if self.family == 'rgalambos':
            return archimedean(family='galambos').get_pdf((1 - u),(1 - v), param)

        elif self.family == 'fgm':
            return 1 + param[0] * (1 - 2 * u) * (1 - 2 * v)

        elif self.family == 'plackett':
            eta = (param[0] - 1)
            term1 = param[0] * (1 + eta * (u + v - 2 * u * v))
            term2 = (1 + eta * (u + v)) ** 2
            term3 = 4 * param[0] * eta * u * v
            return term1 / (term2 - term3) ** (3 / 2)

        elif self.family == 'BB1':
            p0, p1 = param[0], param[1]

            U = -1 + u**(-p1)
            V = -1 + v**(-p1)
            Up = -1 + u**p1
            Vp = -1 + v**p1
            x = U**p0 + V**p0 + 1

            A = x**(1/(p0*p1))
            Bu = Up*u*x
            Bv = Vp*v*x

            pdf = (U**p0)*(V**p0)*(p0*p1 + 1)*(1/A)*(1/Bu)*(1/Bv)

            return pdf

        elif self.family == 'BB2':
            p0, p1 = param

            x = p0/u**p1
            y = p0/v**p1

            logs = scsp.logsumexp([x, y, p0], 0, [1, 1, -1])

            sxy = np.exp(p0 - 0.5*x - 0.5*y)
            s_xdy = np.exp(0.5*x - 0.5*y)
            s_ydx = np.exp(0.5*y - 0.5*x)

            D = (sxy - s_xdy - s_ydx)**2

            A = logs*(logs/p0)**(1/p1)

            pdf = p0**2*u**(-p1 - 1)*v**(-p1 - 1)*(logs*p1 + p1 + 1)/(logs**2*(logs/p0)**(1/p1)*D)

            return pdf

    def LTDC(self, theta):
        """
        # Computes the lower TDC for a given theta

        Parameters
        ----------
        theta : float
            The copula parameter
        """

        if self.family  in ['gumbel', 'joe', 'frank', 'galambos', 'fgm', 'plackett', 'rclayton']:
            return 0

        elif self.family  in ['rgalambos', 'clayton'] :
            return 2 ** (-1 / theta)

        elif self.family  in ['rgumbel', 'rjoe'] :
            return 2 - 2 ** (1 / theta)

    def UTDC(self, theta):
        """
        # Computes the upper TDC for a given theta

        Parameters
        ----------
        theta : float
            The copula parameter
        """

        if self.family  in ['clayton', 'frank', 'fgm', 'plackett', 'rgumbel', 'rjoe', 'rgalambos']:
            return 0

        elif self.family  in ['galambos', 'rclayton'] :
            return 2 ** (-1 / theta)

        elif self.family  in ['gumbel', 'joe'] :
            return 2 - 2 ** (1 / theta)
