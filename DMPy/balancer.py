"""Implementation of Lubitz' parameter balancing as outlined in the paper below.

Lubitz, T., Schulz, M., Klipp, E., & Liebermeister, W. (2010).
    Parameter balancing in kinetic models of cell metabolism.
    The Journal of Physical Chemistry. B, 114(49), 16298-303.
    http://doi.org/10.1021/jp108764b

Author: Rik van Rosmalen
"""
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from numpy.linalg import inv, det
from scipy.linalg import sqrtm

try:
    import optimized_functions
    optimized_functions_available = True
except ImportError:
    optimized_functions_available = False
    warnings.warn("Optimized cython functions not available.")


def cov(sigma):
    """Calculate the covariance matrix from the vector of standard deviations."""
    return np.square(np.diag(sigma))


def shannonEntropy(covariance, dimension=1):
    """Calculate the differential Shannon entropy.

    Based of note 5 of the appendix. Not sure how it is derived.
    Note: Lubitz' code is slightly different, using: 0.5*log(2 * pi * e**2 * det(C))
    """
    return 0.5 * np.log((2 * np.pi * np.e)**dimension * det(covariance))


def base_quantitities(compounds, reactions, stoichiometry):
    """Return the order of the base quantities in a list."""
    # Construct the row layout (base quantities):
    columns = []  # list of (base quantity, compound, reaction)
    # For each reaction: enzyme concentration (u), velocity constant(kV)
    columns.extend([('u', None, r) for r in reactions])
    columns.extend([('kv', None, r) for r in reactions])
    # For each reaction & compounds pair: Michaelis constant (kM) (Ignore kA/kI for now.)
    # for i_s, row in enumerate(stoichiometry):
    #     for i_r, value in enumerate(row):
    #         if value != 0:  # If the compound is involved in the reaction.
    #             columns.append(('km', compounds[i_s], reactions[i_r]))
    for i, j in zip(*np.where(stoichiometry)):  # Faster version
        columns.append(('km', compounds[i], reactions[j]))
    # For each compound: concentration (c), chemical potential (mu)
    columns.extend([('c', s, None) for s in compounds])
    columns.extend([('mu', s, None) for s in compounds])
    return columns


def derived_quantitities(compounds, reactions, stoichiometry):
    """Return the order of the derived quantities in a list."""
    # Construct the row layout
    # (derived quantities, these should be behind the base quantities).
    columns = []  # list of (derived quantity, compound, reaction)
    # For each reaction:
    columns.extend([('keq', None, r) for r in reactions])
    columns.extend([('kcat', None, r) for r in reactions])
    columns.extend([('kcat-', None, r) for r in reactions])
    columns.extend([('vmax', None, r) for r in reactions])
    columns.extend([('A', None, r) for r in reactions])
    # For each compound:
    columns.extend([('mu*', s, None) for s in compounds])
    return columns


def build_priors(base_priors, compounds, reactions, stoichiometry, energy, columns=None):
    """Construct the priors (values, standard deviation) of the base quantities."""
    if columns is None:
        columns = base_quantitities(compounds, reactions, stoichiometry)
    values = np.array([base_priors[p][0] for p, _, _ in columns], dtype=float)
    sigma = np.array([base_priors[p][1] for p, _, _ in columns], dtype=float)
    not_energy = np.array([p not in energy for p, _, _ in columns], dtype=bool)
    return values, sigma, not_energy


def build_default_dependency_matrix(compounds, reactions, stoichiometry, dependencies,
                                    c_pos, r_pos, columns=None):
    """Build the default dependency matrix for forward prediction."""
    if columns is None:
        columns = base_quantitities(compounds, reactions, stoichiometry)
    data = columns[:]
    data.extend(derived_quantitities(compounds, reactions, stoichiometry))
    # Pad the data
    data = [(None, None, p, c, r) for p, c, r in data]
    return build_dependency_matrix(data, compounds, reactions, stoichiometry, dependencies,
                                   c_pos, r_pos, columns=columns)


def build_dependency_matrix(data_in, compounds, reactions, stoichiometry, dependencies,
                            c_pos, r_pos, columns=None):
    """Build the dependency matrix for the input data (backwards predictor)."""
    # Construct the row layout (base quantities):
    if columns is None:
        columns = base_quantitities(compounds, reactions, stoichiometry)
    # Start building the dependency matrix
    Q = np.zeros((len(data_in), len(columns)))

    # For some reason this is a lot more faster in Cython, even without annotation of types.
    if optimized_functions_available:
        return (optimized_functions.build_dependency_matrix(data_in, stoichiometry, dependencies,
                                                            c_pos, r_pos, columns, Q),
                columns)
    else:
        # If it's not available use the plain python version.
        for row, (_, _, parameter, compound, reaction) in enumerate(data_in):
            for column, (dependent_p, dependent_c, dependent_r) in enumerate(columns):
                # Is this dependent and part of the right compound/reaction combo?
                if dependent_p in dependencies[parameter]:
                    d, multiply = dependencies[parameter][dependent_p]
                else:
                    continue

                if (((dependent_c == compound and dependent_r == reaction) or
                     (dependent_c == compound and reaction is None) or
                     (compound is None and dependent_r == reaction))):
                    # If yes, add the dependent term.
                    # d, multiply = dependencies[parameter][columns[column][0]]
                    # print parameter, d, multiply
                    Q[row, column] = d

                if multiply:  # Multiply by stoichiometry?
                    # dependent_p, dependent_c, dependent_r = name
                    # Retrieve correct stoichiometry matrix coordinate
                    # x = compounds.index(compound if compound is not None else dependent_c)
                    # y = reactions.index(reaction if reaction is not None else dependent_r)
                    x = c_pos[compound if compound is not None else dependent_c]
                    y = r_pos[reaction if reaction is not None else dependent_r]
                    m = stoichiometry[x, y]
                    if m:
                        if Q[row, column] == 0:  # Still need to set the base term!
                            Q[row, column] = d
                        Q[row, column] *= m
        return Q, columns


def augment_data(data, priors, compounds, reactions, stoichiometry):
    """Create a list to augment the data with pseudo values for missing values."""
    # The complete list of derived quantities
    derived = derived_quantitities(compounds, reactions, stoichiometry)
    # Create a set to check if we already have something like this
    skip = {(p, c, r) for _, _, p, c, r in data}
    augment = []
    for parameter, compound, reaction in derived:
        if (parameter, compound, reaction) in skip:
            continue
        mean, sd = priors[parameter]
        augment.append((mean, sd, parameter, compound, reaction))
    return augment


def toLog(x, sd, target=None):
    """Convert the values and standard deviation of x to log scale.

    Note: Energies quantities should not be converted to log scale.
    A target array can be used where target is false for non-log scalable values.

    Note:This calculates the arithmetic mean and variance of a logarithmic distribution."""
    # To prevent overflow errors we don't care about, set energy to 1 (in a copy!)
    mx = np.array(x)
    msd = np.array(sd)
    if target is None:
        target = np.ones(mx.shape, dtype=bool)
    else:
        mx[np.logical_not(target)] = 1
        msd[np.logical_not(target)] = 1
    # Calculate log normal values
    logx = np.log(mx) - 0.5 * np.log1p(np.square(msd / mx))
    logvar = np.log1p(np.square(msd / mx))
    logsd = np.sqrt(logvar)
    # Only return the new values where target is True
    return np.where(target, logx, x), np.where(target, logsd, sd)


def toLinearArithmeticMean(logx, logsd, target=None):
    """Convert the values and standard deviation of x to standard linear scale.

    Note: Energies quantities should not be converted to log scale.
    A target array can be used where target is false for non-log scalable values.

    Note: This calculates the arithmetic mean and variance of a logarithmic distribution."""
    # To prevent overflow errors we don't care about, set energy to 1 (in a copy!)
    mlogx = np.array(logx)
    mlogsd = np.array(logsd)
    if target is None:
        target = np.ones(mlogx.shape, dtype=bool)
    else:
        mlogx[np.logical_not(target)] = 1
        mlogsd[np.logical_not(target)] = 1
    # Calculate arithmetic mean and s.d.
    x = np.exp(mlogx + 0.5 * np.square(mlogsd))
    var = np.expm1(np.square(mlogsd)) * np.exp(2 * mlogx + np.square(mlogsd))
    sd = np.sqrt(var)
    # Only return the new values where target is True
    return np.where(target, x, logx), np.where(target, sd, logsd)


def toLinearGeometricMean(logx, target=None):
    """Convert the values and standard deviation of x to standard linear scale.

    Note: Energies quantities should not be converted to log scale.
    A masking array target can be used where target is true for non-energy values.

    Note: This calculates the geometric mean and variance of a logarithmic distribution."""
    # To prevent overflow errors we don't care about, set energy to 1 (in a copy!)
    mlogx = np.array(logx)
    if target is None:
        target = np.ones(mlogx.shape, dtype=bool)
    else:
        mlogx[np.logical_not(target)] = 1
    # Calculate arithmetic mean
    x = np.exp(mlogx)

    # Only return the new values where target is True
    return np.where(target, x, logx)


def balance(compounds, reactions, stoichiometry, data, priors,
            dependencies, nonlog, R, T, augment, c_pos, r_pos):
    """Run the balancing pipeline."""
    S = stoichiometry
    #  Definitions
    # -------------
    # -+- Input
    #     Augment data with priors if no values.
    if augment:
        augment = augment_data(data, priors, compounds, reactions, S)
    else:
        augment = []
    c = base_quantitities(compounds, reactions, S)
    #     Collected kinetic input data
    x_in_prime = np.array([i[0] for i in data + augment])
    #     Collected kinetic input data standard deviation.
    sigma_x_in = np.array([i[1] for i in data + augment])
    if np.any(sigma_x_in == 0):
        warnings.warn("Sigma values of 0 can cause a singular matrix error"
                      " in the inversion of the covariance matrix! Please "
                      "replace with a small but non-zero value.")
    #     Which ones should not be converted to log scale?
    #     - values in nonlog (e.g. energy values)
    not_log = [i[2] not in nonlog for i in data]
    # #     - augmented pseudo values are already in log scale.
    not_log.extend([False for i in augment])
    not_log = np.array(not_log)
    #     Convert to log scale
    x_in_prime, sigma_x_in = toLog(x_in_prime, sigma_x_in, target=not_log)
    # -+- Priors
    #     Prior values and standard deviation of base quantities.
    q_prior, sigma_prior, not_energy_prior = build_priors(priors, compounds,
                                                          reactions,
                                                          S, nonlog, columns=c)
    # q_prior, sigma_prior = toLog(q_prior, sigma_prior, target=not_energy_prior)
    # -+- Dependency matrices
    #     Forward predictor (Prediction matrix)
    Q, Q_columns = build_default_dependency_matrix(compounds, reactions,
                                                   S, dependencies, c_pos, r_pos,
                                                   columns=c)
    #     Backwars predictor (Data dependency matrix)
    Q_prime, Q_prime_columns = build_dependency_matrix(data + augment, compounds,
                                                       reactions, S,
                                                       dependencies, c_pos,
                                                       r_pos, columns=c)
    #  Calculations
    # --------------
    #   Covariance matrix of sigma x
    C_x = cov(sigma_x_in)
    #   Covariance matrix of prior base quantities.
    C_prior = cov(sigma_prior)
    #   Posterior covariance matrix (base quantities)
    #   Posterior parameter distribution (base quantities)
    iC_prior = inv(C_prior)
    iC_x = inv(C_x)
    try:
        C_post = inv(iC_prior + Q_prime.T.dot(iC_x).dot(Q_prime))
        q_post = C_post.dot(Q_prime.T.dot(iC_x).dot(x_in_prime) +
                            iC_prior.dot(q_prior))
    except np.linalg.LinAlgError:
        # The inversion of C_post might fail because of a singular matrix error.
        # We can avoid this by using left matrix division. See note 4 (p. 6) of
        # the appendix of Lubitz' balancing paper.
        a = iC_prior + Q_prime.T.dot(iC_x).dot(Q_prime)
        b = (Q_prime.T.dot(iC_x).dot(x_in_prime) + iC_prior.dot(q_prior))
        q_post = np.linalg.lstsq(a, b)[0]
    #   Posterior means (all quantities)
    # x_post = Q.dot(q_post)
    #   Posterior covariance (all quantities)
    try:
        C_x_post = Q.dot(C_post).dot(Q.T)
    except UnboundLocalError:
        # If we couldn't calculate the inverse before, C_post is sill undefined.
        C_x_post = np.linalg.lstsq(Q.T, a.T)[0].dot(Q.T)

    # Note: Lubitz' script appears to be able to do some optimization step for the bounds as well.
    #    Results
    # -------------
    #   Get standard deviation of the posterior
    # sigma_post = np.sqrt(np.diag(C_x_post))
    #   Get columns of result
    #   Note: These are the rows of Q!
    x_post_columns = (base_quantitities(compounds, reactions, S) +
                      derived_quantitities(compounds, reactions, S))
    #   Revert back to log scale
    # not_log_post = np.array([(i[0] not in nonlog) for i in x_post_columns])
    # x_post, sigma_post = toLinearGeometricMean(x_post, sigma_post, target=not_log_post)
    # x_post, sigma_post = toLinearArithmeticMean(x_post, sigma_post, target=not_log_post)

    # Note that since we actually do need C_post, we will still crash here.
    try:
        C_post
    except UnboundLocalError:
        raise ValueError("Couldn't calculate C_post because of singular matrix.")

    return BalancingResult(Q, q_post, C_post, C_x_post, x_post_columns)


class BalancingResult(object):
    """Results class for balancing that allows easy re-sampling."""

    def __init__(self, Q, q_post, C_post, C_x_post, columns, not_log=None):
        """Create a result object from balancing result allowing for sampling."""
        self.Q = Q
        self.q_post = q_post
        self.C_post = C_post
        self.C_x_post = C_x_post
        self.columns = columns

        if not_log is None:
            self.not_log = np.array([(i[0] not in nonlog) for i in self.columns])
        else:
            self.not_log = not_log

        # Lazy load these (might be large matrix operations!)
        self._mean = None
        self._median = None
        self._sd = None
        self._A = None

    @property
    def median(self):
        """Calculate median values for each parameter distribution using the geometric mean."""
        if self._median is None:
            self._median = toLinearGeometricMean(self.Q.dot(self.q_post),
                                                 target=self.not_log)
        return self._median

    @property
    def mean(self):
        """Calculate median values for each parameter distribution using the geometric mean."""
        if self._mean is None:
            sigma = np.sqrt(np.diag(self.C_x_post))
            self._mean, self._sd = toLinearArithmeticMean(self.Q.dot(self.q_post),
                                                          sigma, target=self.not_log)
        return self._mean

    @property
    def sd(self):
        """Calculate standard deviations for each parameter distribution."""
        if self._sd is None:
            # Calculating the mean property will also calculate the hidden sd property.
            self.mean
        return self._sd

    def sample(self, normal=None, tolerance=1e-12):
        """Draw a sample from the final distribution using a (normal) error distribution."""
        if self._A is None:
            self._A = sqrtm(self.C_post)
        if normal is None:
            normal = np.random.normal(0, 1, self.q_post.shape)

        # TODO: Consider an implementation using np.random.multivariate_normal?
        sample = self.Q.dot(self.q_post + self._A.dot(normal))
        if np.any(sample.imag > tolerance):
            raise ValueError("Imaginary parts in matrix square root of C_post larger than "
                             "tolerance: {} > {}".format(sample.imag.max(), tolerance))
        # Take the exponent of everything not-energy.
        return toLinearGeometricMean(sample.real, self.not_log)

    def save(self, path):
        """Save the results to a file so they can be loaded later."""
        arrays = {'Q': self.Q,
                  'q_post': self.q_post,
                  'C_post': self.C_post,
                  'C_x_post': self.C_x_post,
                  'columns': self.columns,
                  'not_log': self.not_log}
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Load the results to a file previously saved."""
        x = np.load(path, allow_pickle=True)
        Q = x['Q']
        q_post = x['q_post']
        C_post = x['C_post']
        C_x_post = x['C_x_post']
        columns = x['columns']
        not_log = x['not_log']
        return cls(Q, q_post, C_post, C_x_post, columns.tolist(), not_log)


# Note: Old version
def print_balancing_results(data, x_post, sigma_post, x_post_columns):
    """Print a neat table with the new and old parameter values."""
    print(' type - Reaction ( Compound) |      old      (sd) >>      new       (sd)')
    print('-' * 78)
    for i, j in enumerate(x_post_columns):
        print("{0:>5} - {2:>9}({1:>9}) |".format(*(a if a is not None else '.' for a in j)),
              end='')
        p, c, r = j
        for v, sd, p2, c2, r2 in data:
            if p == p2 and c == c2 and r == r2:
                print('{:>8.1e} ({:>7.1e})'.format(v, sd), end='')
                break
        else:
            print('                  ', end='')
        print(">> {:>11.3e} ({:>11.3e})".format(x_post[i], sigma_post[i]))


# Note: Old version
def save_balancing_results(x_post, sigma_post, x_post_columns, filepath):
    """Save the results to a .tsv file in path."""
    p_dict = {'km': ('Michaelis constant', 'mM'),
              'ki': ('inhibitory constant', 'mM'),
              'ka': ('activation constant', 'mM'),
              'keq': ('equilibrium constant', ''),
              'kv': ('catalytic rate constant geometric mean', '1/s'),
              'u': ('concentration of enzyme', 'mM'),
              'kcat': ('substrate catalytic rate constant', '1/s'),
              'kcat-': ('product catalytic rate constant', '1/s'),
              'vmax': ('forward maximal velocity', 'mM/s'),
              'A': ('reaction affinity', 'kJ/mol'),
              'mu*': ('standard chemical potential', 'kJ/mol'),
              'mu': ('chemical potential', 'kJ/mol'),
              'c': ('concentration', 'mM')
              }

    with open(filepath, 'w') as outf:
        outf.write('QuantityType\tSBMLReactionID\tSBMLSpeciesID\tValue\tMean\t'
                   'Std\tUnit\tProvenance\tType\tSource\tlogMean\tlogStd\t'
                   'Temperature\tpH\tMinimum\tMaximum\n')
        for (p, c, r), x, s in zip(x_post_columns, x_post, sigma_post):
            p, u = p_dict[p]
            if r is None:
                r = ''
            if c is None:
                c = ''
            strings = [str(i) for i in (p, r, c, x, x, s, u, '', '', '', '', '', '', '', '', '')]
            outf.write('\t'.join(strings))
            outf.write('\n')


# ------------------------------------------------------
# The parameters below can be changed.
# Everything here is using the values from Lubitz et al.
# ------------------------------------------------------
# Some constants.
R = 8.314 / 1000.  # 8.314 J/(mol*K) => J/(millimol * K)
T = 300.
# True or False means that it should/shouldn't be multiplied with the stoichiometry.
dependencies = {
    'c':
        {'c': (1.0, False),
         },
    'km':
        {'km': (1.0, False),
         },
    'vmax':
        {'mu': (-1.0 / (2.0 * R * T), True),
         'kv': (1.0, False),
         'km': (-1.0 / 2.0, True),
         'u': (1.0, False),
         },
    'kcat':
        {'mu': (-1.0 / (2.0 * R * T), True),
         'kv': (1.0, False),
         'km': (-1.0 / 2.0, True),
         },
    'kcat-':
        {'mu': (1.0 / (2.0 * R * T), True),
         'kv': (1.0, False),
         'km': (1.0 / 2.0, True),
         },
    'keq':
        {'mu': (-1.0 / (R * T), True),
         },
    'u':
        {'u': (1.0, False),
         },
    'kv':
        {'kv': (1.0, False),
         },
    'mu':
        {'mu': (1.0, False),
         },
    'mu*':
        {'mu': (1.0, False),
         'c': (R * T, False),
         },
    'A':
        {'mu': (-1.0, True),
         'c': (-R * T, True),
         },
    'ka':
        {'ka': (1.0, False),
         },
    'ki':
        {'ki': (1.0, False),
         },
    }
# Properties that should not be converted log-scale
# (example: Energy as it can be negative).
nonlog = set(['mu', 'mu*', 'A'])
# Priors - extracted from lubitz' script
# These are the values in linear(median), log10(sd) scale.
priors = {
          # Base quantities
          'mu': (-880.0,   680.0),
          'kv': (  10.0,     1.0),
          'km': (   0.1,     1.0),
          'c':  (   0.1,     1.5),
          'u':  (   0.00001, 1.5),
          'ki': (   0.1,     1.0),
          'ka': (   0.1,     1.0),
          # Derived quantities
          'keq':   (   1.0,   1.5),
          'kcat':  (  10.0,   1.5),
          'kcat-': (  10.0,   1.5),
          'vmax':  (   0.001, 2.0),
          'A':     (   0.0,  10.0),
          'mu*':   (-880.0, 680.0),
         }
# Same values in linear, linear scale
priors = {
          # Base quantities
          'mu': (-880.0,   680.00),
          'kv': (  10.0,     6.26),
          'km': (   0.1,     6.26),
          'c':  (   0.1,    10.32),
          'u':  (   0.0001, 10.32),
          'ki': (   0.1,     6.26),
          'ka': (   0.1,     6.26),
          # Derived quantities
          'keq':   (   1.0,   10.32),
          'kcat':  (  10.0,   10.32),
          'kcat-': (  10.0,   10.32),
          'vmax':  (   0.001, 17.01),
          'A':     (   0.0,   10.00),
          'mu*':   (-880.0,  680.00),
         }
# Custom priors, less spread for better results.
priors = {
          # Base quantities
          'mu': (-880.0,  680.00),
          'kv': (  10.0,    2.00),
          'km': (  0.01,    1.20),
          'c':  (   0.1,    2.00),
          'u':  (   1.0,    1.60),
          'ki': (   0.1,    2.00),
          'ka': (   0.1,    2.00),
          # Derived quantities
          'keq':   (   1.0,   2.00),
          'kcat':  (   1.0,   2.00),
          'kcat-': (   1.0,   2.00),
          'vmax':  (  10.0,   2.00),
          'A':     (   0.0,  10.00),
          'mu*':   (-880.0, 680.00),
         }
# priors = {
#           'mu':  (-880.0000,   4.0),
#           'kv':    (10.0,   1000.0),
#           'km':     (0.1,     10.0),
#           'c':      (0.1,     10.0),
#           'u':      (0.0001,   0.01),
#           'ki':     (0.1,     10.0),
#           'ka':     (0.1,     10.0),
#           'keq':    (1.0,    100.0),
#           'kcat':  (10.0,   1000.0),
#           'kcat-': (10.0,   1000.0),
#           'vmax':   (0.001,    0.1),
#           'A':    (  0.0,     10.0),
#           'mu*': (-880.0,      4.0),
#           }
# Convert priors to proper log scale
# Means are given in base_priors as linear
# Standard deviations are given in log10
for key, value in priors.items():
    if key not in nonlog:
        # This is for lin(median), log10(sd)
        # priors[key] = (np.log(value[0]), value[1] * np.log(10))
        # This is for lin, lin scale
        priors[key] = (np.log(value[0]), np.log(value[1]))


if __name__ == "__main__":
    # -----------------
    # Minimal test case
    # -----------------
    compounds = ['G6P', 'F6P']
    reactions = ['PGI']
    data = [
            (10.0, 1.0, 'c', 'G6P', None),
            (10.0, 1.0, 'c', 'F6P', None),
            (0.28, 0.056, 'km', 'G6P', 'PGI'),
            (0.147, 0.0294, 'km', 'F6P', 'PGI'),
            (0.361, 0.0361, 'keq', None, 'PGI'),
            (1511, 151, 'vmax', None, 'PGI'),
            ]
    S = np.zeros((len(compounds), len(reactions)))
    S[:, 0] = [-1,  1]

    # ---------------
    # Small test case
    # ---------------
    # compounds = ['Glc', 'ATP', 'G6P', 'ADP']
    # reactions = ['vHK']
    # data = []
    # S = np.zeros((len(compounds), len(reactions)))
    # S[:, 0] = [-1,  -1,  1,  1]

    # -------------------------------------
    # Bigger test case with some input data
    # -------------------------------------
    # compounds = ['A', 'B', 'C', 'D', 'E']
    # compounds = ['A', 'B', 'C', 'D', 'E', 'F']
    # reactions = ['X', 'Y', 'Z']
    # data = [
            # (1.0, 0.1,    'c',  'A', None),
            # (9.0, 0.001, 'c',  'B', None),
            # (1.2, 0.1,   'km',  'A',  'X'),
            # (0.2, 0.1,   'km',  'A',  'Y'),
            # (1.2, 0.2, 'vmax',  'A',  'X'),
            # (2.2, 0.2, 'kcat',  'A',  'Y'),
            # (3.2, 0.2,   'km',  'C',  'Z'),
            # (5.0, 0.001,  'keq', None,  'Z'),
            # (0.1, 0.1,  'keq', None,  'X')
            # ]

    # X: A + A => B
    # Y: B + A => C
    # Z: C     => D
    # S = np.zeros((len(compounds), len(reactions)))
    # S[:, 0] = [-2,  1,  1,  0, 0]
    # S[:, 1] = [-1, -1,  1,  0, 0]
    # S[:, 2] = [ 0,  0, -1,  1, 0]

    # S[:, 0] = [-1,  1,  0,  0,  0,  0]
    # S[:, 1] = [ 0,  0, -1,  1,  0,  0]
    # S[:, 2] = [ 0,  0,  0,  0, -1,  1]
    s_pos = dict([(j, i) for i, j in enumerate(compounds)])
    r_pos = dict([(j, i) for i, j in enumerate(reactions)])
    result = balance(compounds, reactions,
                     S, data, priors,
                     dependencies, nonlog,
                     R, T, True, s_pos, r_pos)
    print_balancing_results(data, result.median, result.sd, result.columns)

    # --------------------
    # Costa model test case
    # --------------------
    # import sbmlwrap

    # parameters = {
    #               'equilibrium constant': 'keq',
    #               'substrate catalytic rate constant': 'kcat',
    #               'Michaelis constant': 'km',
    #               'concentration': 'c',
    #               'inhibitory constant': 'ki',
    #               'forward maximal velocity': 'vmax',
    #               }

    # # m = sbmlwrap.Model('data/Hynne2001-fixedSto.xml')
    # m = sbmlwrap.Model('output-coli-core/coli_core_compressed.xml')
    # S = sbml_model.get_stoichiometry_matrix()
    # r_pos = dict(((j, i) for i, j in enumerate(sbml_model.reactions.keys())))
    # s_pos = dict(((j, i )for i, j in enumerate(sbml_model.species.keys())))

    # compounds = list(sbml_model.species.keys())
    # reactions = list(sbml_model.reactions.keys())

    # data = []
    # with open('output-coli-core/parametersbesrm.tsv', 'r') as datafile:
    # # with open('output-costa/parameters.tsv', 'r') as datafile:
    #     datafile.readline()  # Skip header
    #     for line in datafile:
    #         line = line.split('\t')
    #         if line[3] == 'nan':
    #             continue
    #         v = float(line[3])
    #         if v < 1E-12:
    #             v = 1E-12
    #         sd = 0.1 * v if line[4] == 'None' else float(line[4])
    #         p = parameters[line[0]]
    #         c = None if (line[2] == 'None' or line[2] == 'nan') else line[2]
    #         r = None if (line[1] == 'None' or line[1] == 'nan') else line[1]
    #         if (c in s_pos or c is None) and (r in r_pos or r is None):
    #             data.append((v, sd, p, c, r))
    # with open('data/costa_parameters.csv', 'r') as datafile:
    #     datafile.readline()  # Skip header
    #     for line in datafile:
    #         line = line.split('\t')
    #         if line[3] == 'nan':
    #             continue
    #         v = float(line[3])
    #         if v < 1E-12:
    #             v = 1E-12
    #         sd = 0.001 * v #if line[4] == 'None' else float(line[4])
    #         p = parameters[line[0]]
    #         c = None if (line[2] == 'None' or line[2] == 'nan' or not line[2]) else line[2]
    #         r = None if (line[1] == 'None' or line[1] == 'nan' or not line[1]) else line[1]
    #         # if p not in ("km"):
    #             # print p, r
    #             # continue
    #         data.append((v, sd, p, c, r))

    # --------------------
    # Big test case
    # --------------------
    # import sbmlwrap

    # m = sbmlwrap.Model('data/iJO1366.xml')
    # S = sbml_model.get_stoichiometry_matrix()
    # r_pos = dict(((j, i) for i, j in enumerate(sbml_model.reactions.keys())))
    # s_pos = dict(((j, i )for i, j in enumerate(sbml_model.species.keys())))

    # compounds = list(sbml_model.species.keys())
    # reactions = list(sbml_model.reactions.keys())
    # data = []

    # --------------------
    # Intermediate test case
    # --------------------
    # n = 1000

    # compounds = ['C' + str(i) for i in range(n)]
    # s_pos = dict(zip(compounds, range(n)))
    # reactions = ['R' + str(i) for i in range(n)]
    # r_pos = dict(zip(reactions, range(n)))
    # S = np.random.random(((len(compounds), len(reactions)))) > 0.99
    # data = []

    # --------------------
    # Run test case
    # --------------------
    x_post, sigma_post, x_post_columns = balance(compounds, reactions, S, data,
                                                 priors, dependencies,
                                                 nonlog, R, T, True, s_pos, r_pos)
    # result = balance(compounds, reactions,
    #                  S, data, priors,
    #                  dependencies, nonlog,
    #                  R, T, True, s_pos, r_pos)
    # print_balancing_results(data, result.median, result.sd, result.columns)

    # -------------------------------
    # Check naive sampling vs. proper
    # -------------------------------
    # proper = []
    # naive = []

    # for i in range(10000):
    #     normal = np.random.normal(0, 1, result.median.size)
    #     short_normal = normal[:result.q_post.size]
    #     proper.append(result.sample(short_normal))
    #     naive.append(result.median + normal * result.sd)

    # proper = np.array(proper)
    # naive = np.array(naive)

    # minima = np.fmin(proper.min(axis=0), naive.min(axis=0))
    # maxima = np.fmax(proper.max(axis=0), naive.max(axis=0))
    # import matplotlib.pyplot as plt

    # for idx in range(proper.shape[1]):
    #     plt.figure()
    #     plt.title(' '.join(i for i in result.columns[idx] if i is not None))
    #     bins = np.linspace(minima[idx], maxima[idx])
    #     plt.hist(naive[:, idx], bins, alpha=.4, color='b')
    #     plt.hist(proper[:, idx], bins, alpha=.4, color='g')
    #     # Non energy quantities should have a log normal distribution
    #     if result.columns[idx][0] not in nonlog:
    #         plt.xscale('log')
    # plt.show()
