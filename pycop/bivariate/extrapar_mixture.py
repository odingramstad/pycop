import numpy as np
from pycop.bivariate.copula import copula
from pycop.bivariate.archimedean import archimedean
from pycop.bivariate.gaussian import gaussian


class extrapar_mixture(copula):
    """
    # Creates an extrapar mixture copula objects

    ...

    Attributes
    ----------
    dim : int
        The number of copula combined, only 2 or 3 supported
    mixture_type : str
        The type of mixture as bundle of the combinated copula
    cop : list
        A list that contains the copula objects to combine
    bounds_param : list
        A list that contains the domain of the parameter(s) in a tuple.
        Example : [(lower, upper), (lower, upper)]
    parameters_start : array
        Value(s) of the initial guess when estimating the copula parameter(s).
        It represents the parameter `x0` in the `scipy.optimize.minimize` function.

    Methods
    -------
    get_cdf(u, v, param)
        Computes the Cumulative Distribution Function (CDF).
    get_pdf(u, v, param)
        Computes the Probability Density Function (PDF).
    LTDC(w1, theta1)
        Computes the Lower Tail Dependence Coefficient (TDC).
    UTDC(w1, theta2)
        Computes the upper TDC.
    """

    def __init__(self, copula_list):
        """
        Parameters
        ----------
        copula_list : list
            A list of string of the type of copula to combined
            Example : ["clayton", "gumbel"]

        Raises
        ------
        ValueError
            dim : the lenght of `copula_list` must be equal to 2 or 3.
            Mixtures are only available for a combination of 2 or 3 copulas

            copula_list : element must be supported functions.
            Mixtures are only available for archimedean and gaussian.

        """
        # the `student` copula class inherit the `copula` class
        super().__init__()

        Archimedean_families = [
            'clayton', 'gumbel', 'frank', 'joe', 'galambos','fgm', 'plackett',
            'rgumbel', 'rclayton', 'rjoe','rgalambos']

        self.dim = len(copula_list)

        if self.dim != 2:
            print("Mixture supported only for combinaison of 2 copulas")
            raise ValueError

        self.cop = []
        mixture_type = copula_list[0].capitalize()

        for cop in copula_list[1:]:
            mixture_type+= "-"+cop.capitalize()

        self.family = mixture_type + " mixture"
        self.bounds_param = [(0, 1), (0, 1)]
        self.parameters_start = [np.array(0.5), np.array(0.5)]

        for i in range(0, self.dim):
            if copula_list[i] == "gaussian":
                self.cop.append(gaussian())
                self.bounds_param.append((-1, 1))
                self.parameters_start.append(np.array(0))

            elif copula_list[i] in Archimedean_families:
                cop_mixt = archimedean(family=copula_list[i])
                self.cop.append(cop_mixt)
                self.bounds_param.append(cop_mixt.bounds_param[0])
                self.parameters_start.append(cop_mixt.parameters_start)
            else:
                print("Mixture only supported for archimedean and gaussian mixture only")
                print("Archimedean copula available are: ", Archimedean_families)
                raise ValueError
        self.parameters_start = tuple(self.parameters_start)

    def get_cdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            A list that contains the parameters of the mixture and the copula.
            The element of the list must be ordered as follow, for 2-dimensional mixture :
                [
                    weight1 : float, weight1 ∈ [-1,1]
                        The weight given in the first copula.
                    theta1 : float
                        The theta parameter of the first copula.
                    theta2 : float
                        " second.
                ]
            For a 3-dimensional mixture :
                [
                    weight1 : float
                        the weight given in the first copula.
                    weight2 : float
                        " second.
                    weight3 : float
                        " third.
                    theta1 : float
                        The theta parameter of the first copula.
                    theta2 : float
                        " second.
                    theta3 : float
                        " third.
                ]
            The sum of the weights must be equal to 1.
        """
        a, b = param[:2]

        u_a = u**a
        v_b = v**b
        u_1_a = u**(1-a)
        v_1_b = v**(1-b)

        np0 = len(self.cop[0].bounds_param)
        np1 = len(self.cop[1].bounds_param)

        A_params = param[2:2+np0]
        B_params = param[2+np0:2+np0+np1]

        return self.cop[0].get_cdf(u_a, v_b, A_params)*self.cop[1].get_cdf(u_1_a, v_1_b, B_params)

    def get_pdf(self, u, v, param):
        """
        # Computes the CDF

        Parameters
        ----------
        u, v : float
            Values of the marginal CDFs
        param : list
            A list that contains the parameters of the mixture and the copula.
            See how to order the list in the above method `get_cdf`
        """

        a, b = param[:2]

        u_a = u**a
        v_b = v**b
        u_1_a = u**(1-a)
        v_1_b = v**(1-b)

        A = self.cop[0]
        B = self.cop[1]

        np0 = len(A.bounds_param)
        np1 = len(B.bounds_param)

        A_params = param[2:2+np0]
        B_params = param[2+np0:2+np0+np1]

        cdf_u_A, cdf_v_A = A.get_grad_cdf(u_a, v_b, A_params)
        cdf_u_B, cdf_v_B = B.get_grad_cdf(u_1_a, v_1_b, B_params)

        part1 = a*b*B.get_cdf(u_1_a, v_1_b, B_params)*A.get_pdf(u_a, v_b, A_params)/(u_1_a*v_1_b)
        part2 = (1-a)*b*cdf_v_A*cdf_u_B/(u_a*v_1_b)
        part3 = a*(1-b)*cdf_u_A*cdf_v_B/(u_1_a*v_b)
        part4 = (1-a)*(1-b)*A.get_cdf(u_a, v_b, A_params)*B.get_pdf(u_1_a, v_1_b, B_params)/(u_a*v_b)
        return part1 + part2 + part3 + part4

    def LTDC(self, theta, weight):
        """
        # Computes the upper TDC

        Parameters
        ----------
        weight : float
            The weight associated to the copula with Lower Tail Dependence
        theta : float
            The parameter of the copula with Lower Tail Dependence
        """
        return self.cop[0].LTDC(theta)*weight

    def UTDC(self, theta, weight):
        """
        # Computes the upper TDC

        Parameters
        ----------
        weight : float
            The weight associated to the copula with Upper Tail Dependence
        theta : float
            The parameter of the copula with Upper Tail Dependence
        """
        return self.cop[-1].UTDC(theta)*weight
