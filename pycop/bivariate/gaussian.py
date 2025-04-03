import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.special import erfinv
from pycop.bivariate.copula import copula

class gaussian(copula):
    """
    # Creates a gaussian copula object

    ...

    Attributes
    ----------
    family : str
        = "gaussian"
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
    """

    def __init__(self):
        # the `gaussian` copula class inherit the `copula` class
        super().__init__()
        self.family = "gaussian"
        self.bounds_param = [(-1, 1)]
        self.parameters_start = np.array(0)

    def get_cdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            The correlation coefficient param[0] ∈ [-1,1].
            Used to defined the correlation matrix (squared, symetric and definite positive)
        """

        y1 = norm.ppf(u, 0, 1)
        y2 = norm.ppf(v, 0, 1)
        rho = param[0]

        return multivariate_normal.cdf(np.array([y1, y2]).T, mean=None, cov=[[1, rho], [rho, 1]])

    def get_dcdf_duv(self, deriv, u, v, param):
        """
        # Computes the partial derivative of the CDF

        Parameters
        ----------
        deriv : string
             Which derivative to take: 'u' or 'v'.
        u, v : float
            Values of the marginal CDFs
        param : list
            The correlation coefficient param[0] ∈ [-1,1].
            Used to defined the correlation matrix (squared, symetric and definite positive)
        """
        y1 = norm.ppf(u, 0, 1)
        y2 = norm.ppf(v, 0, 1)

        pdf_y1 = norm.pdf(y1)
        pdf_y2 = norm.pdf(y2)

        dy1_du = 1/pdf_y1
        dy2_dv = 1/pdf_y2

        rho = param[0]

        if deriv == 'u':
            yt = (y2 - rho*y1)/np.sqrt(1 - rho**2)
            return dy1_du*norm.pdf(y1)*norm.cdf(yt)
        elif deriv == 'v':
            yt = (y1 - rho*y2)/np.sqrt(1 - rho**2)
            return dy2_dv*norm.pdf(y2)*norm.cdf(yt)

    def get_pdf(self, u, v, param):
        """
        # Computes the PDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            The correlation coefficient param[0] ∈ [-1,1].
            Used to defined the correlation matrix (squared, symetric and definite positive)
        """

        rho = param[0]
        a = np.sqrt(2) * erfinv(2 * u - 1)
        b = np.sqrt(2) * erfinv(2 * v - 1)
        det_rho = 1 - rho**2

        return det_rho**-0.5 * np.exp(-((a**2 + b**2) * rho**2 -2 * a * b * rho) / (2 * det_rho))
