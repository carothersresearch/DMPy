#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to generate random metabolic models and their parameters for use in the pipeline.

Author: Rik van Rosmalen

A metabolic network consists of a hypergraph with nodes (metabolites) and edges (reactions),
where each edge can have multiple incoming and outgoing nodes.
The hypergraph can be converted to a bipartite graph, consisting of two sets of nodes (metabolites
and reactions), and edges denote the participation of a metabolite in a reaction, and thus always
cross the node group boundaries.

To properly generate randomized metabolic networks, many properties have to be considered (See
citations below):
    - Scale free (power-law) degree distribution
    - Large clustering coefficients
    - Small average path length
    - degree-degree correlation
    - centrality measures
    - Distribution and over-representation of sub-network motifs
Additional considerations can be made for the following:
    - Mass balance constraints
    - Thermodynamic feasibility [directionality]
    - Distribution of kinetic properties

Note that not all of these properties might be essential depending on your usage case of the
generated networks.

"Nevertheless, this direction in network research has resulted in the discovery of
salient properties of biological networks, i.e. properties which show similar trends
for a wide variety of networks from different cells, tissues and species.
Some of these properties include: scale-free (i.e. power-law) degree distribution,
large clustering coefficient, small average path length, degree–degree correlation,
different behavior of various centrality measures and the distribution and overrepresentation
of subnetworks, known as motifs (Barabási and Oltvai, 2004; Milo et al., 2002)."
By: Basler, G., Ebenhöh, O., Selbig, J., & Nikoloski, Z. (2011).
    Mass-balanced randomization of metabolic networks.
    Bioinformatics, 27(10), 1397–1403.
    http://doi.org/10.1093/bioinformatics/btr145

"Previous studies have revealed that metabolic networks of living organisms are highly structured.
For example, the degree distribution of the metabolites in these networks has a power law tail
[10,11]. Metabolic networks seem to have further remarkable features such as a high level of
clustering [12]."
By: Samal, A., & Martin, O. C. (2011).
    Randomizing Genome-Scale Metabolic Networks.
    PLoS ONE, 6(7), e22295.
    http://doi.org/10.1371/journal.pone.0022295


"""
from __future__ import print_function
from __future__ import division

import collections
import warnings

import numpy as np
import scipy.stats

import libsbml

import networkx as nx

import balancer
import sbmlwrap


# Not really sure about the exact applicability of all types.
# The SBO terms are not very consistent
SBO_regulation = {
                  # Relevant inhibitor types
                  20: 'inhibitor',  # simple inhibition
                      206: 'competitive inhibitor',  # specific inhibition
                      207: 'non-competitive inhibitor',
                          536: 'partial inhibitor',  # partial inhibition
                          537: 'complete inhibitor',  # complete inhibition
                      639: 'allosteric inhibitor',  # partial inhibition
                  # Relevant activator types
                  459: 'stimulator',
                      461: 'essential activator',
                          535: 'binding activator',  # N.A. - changes km
                          534: 'catalytic activator',  # N.A. - changes kcat
                          533: 'specific activator',  # specific activation
                      462: 'non-essential activator',  # simple activation
                      21: 'potentiator',
                          636: 'allosteric activator',  # complete activation
                          637: 'non-allosteric activator',  # partial activation
                  }

# These SBO terms are assumed to represent the following regulation types.
SBO_kinetics = {
                 20:  ('inhibition', 'simple'),
                 206: ('inhibition', 'specific'),
                 537: ('inhibition', 'complete'),
                 536: ('inhibition', 'partial'),

                 462: ('activation', 'simple'),
                 533: ('activation', 'specific'),
                 636: ('activation', 'complete'),
                 637: ('activation', 'partial'),
                 }

SBO_inhibitors = {20, 206, 207, 536, 537, 639}
SBO_activators = {459, 461, 535, 534, 533, 462, 21, 636, 637}


class UID(object):
    """
    Unique IDentifier object.

    Generates an increasing series of numbers that can be used as unique identifiers.
    Can be used without instantiating an object through the get method.
    If a new object is instantiated, it will count separately starting from 0.
    """
    current = 0

    def __init__(self):
        self.current = 0

        def get(self):
            self.current += 1
            return self.current
        from types import MethodType
        self.get = MethodType(get, self)

    @classmethod
    def get(cls):
        cls.current += 1
        return cls.current


class Network(object):
    N_UID = UID()

    def __init__(self, name=None, metabolites=None, reactions=None, compartments=None,
                 parameter_distributions=None):
        """Create an empty Network object."""
        self.name = name if name is not None else 'N_{}'.format(self.N_UID.get())
        self.metabolites = set() if metabolites is None else metabolites
        self.reactions = set() if reactions is None else reactions
        self.compartments = set() if compartments is None else compartments
        self.parameter_distributions = ({} if parameter_distributions is None
                                        else parameter_distributions)
        self.distributions = {}

    @classmethod
    def from_sbml_file(cls, file):
        """Create a Network from an SBML file."""
        with open(file) as sbml_file:
            return Network.from_sbml_string(sbml_file.read())

    @classmethod
    def from_sbml_string(cls, string):
        """Create a Network from an SBML string."""
        model = sbmlwrap.Model.from_string(string)
        compartments, metabolites, reactions = {}, {}, {}
        for c_id, compartment in model.compartments.items():
            volume = compartment.size
            compartments[c_id] = Compartment(c_id, volume=volume)

        for m_id, metabolite in model.species.items():
            m = Metabolite(m_id)
            metabolites[m_id] = m
            compartments[metabolite.compartment.id].metabolites.add(m)

        for r_id, reaction in model.reactions.items():
            r = Reaction(r_id)
            reactions[r_id] = r
            for m, s in reaction.reactants:
                r.reactants[metabolites[m.id]] += s
            for m, s in reaction.products:
                r.products[metabolites[m.id]] += s
            for m in reaction.modifiers:
                # Get the SBO term if available
                sbml_obj = reaction.sbml_object.getModifier(m.id)
                if sbml_obj.isSetSBOTerm():
                    r.regulators[metabolites[m.id]] = sbml_obj.getSBOTerm()
                else:
                    r.regulators[metabolites[m.id]] = 0

        return cls(metabolites=set(metabolites.values()), reactions=set(reactions.values()),
                   compartments=set(compartments.values()))

    @classmethod
    def from_stoichiometry(cls, S, C=None, R=None, C_volumes=None):
        """Create a network from a stoichiometry matrix of n_metabolites x n_reactions.

        In the matrix S[n_metabolites, n_reactions], the value i[n_m, n_r] is the stoichiometry
        of the metabolite m for the reaction r.
        The array C denotes in which compartment each metabolite resides. If not provided, it will
        be assumed that everything is in the same compartment.
        """
        n_metabolites, n_reactions = S.shape
        if C is None:
            C = np.zeros(n_metabolites, dtype=int)
        if C_volumes is None:
            C_volumes = np.ones(np.unique(C).size, dtype=float)
        if R is None:
            R = np.zeros(S.shape, dtype=int)

        self = cls()

        # Create metabolites, reactions and compartments.
        metabolites, reactions, compartments = [], [], []
        for i in range(C.max() + 1):
            compartments.append(Compartment(volume=C_volumes[i]))
        for i in range(n_metabolites):
            m = Metabolite()
            metabolites.append(m)
            compartments[C[i]].metabolites.add(m)
        for i in range(n_reactions):
            reactions.append(Reaction())

        # Connect according to stoichiometry
        edges = S != 0
        for s, idx in zip(S[edges], np.argwhere(edges)):
            m, r = idx
            if s > 0:
                reactions[r].products[metabolites[m]] += s
            elif s < 0:
                reactions[r].reactants[metabolites[m]] -= s

        # Add regulators
        for m, r in np.argwhere(R != 0):
            reactions[r].regulators[metabolites[m]] = R[m, r]

        self.metabolites = set(metabolites)
        self.reactions = set(reactions)
        self.compartments = set(compartments)
        return self

    @classmethod
    def from_networkx_graph(cls, graph):
        """Instantiate a network from a NetworkX graph."""
        self = cls()

        compartments = {}
        for name, volume in G.graph['compartments'].items():
            compartments[name] = Compartment(name, volume=volume)

        reactions_to_fix = []
        add_regulators = []
        # If we need to build the objects, we will need a reference dictionary.
        metabolites = {}
        for node, data in graph.nodes(data=True):
            if data['bipartite'] == 1:
                # Check if the nodes are actual Reaction objects, else we reconstruct them.
                if isinstance(node, Reaction):
                    self.reactions.add(node)
                else:
                    r = Reaction()
                    # We still need to add the metabolites, but we will do that later since
                    # they might not be reconstructed yet.
                    reactions_to_fix.append((node, r))
                    self.reactions.add(r)
                    if 'regulators' in data:
                        add_regulators.append((r, data['regulators']))
            if data['bipartite'] == 0:
                # Same deal as for the reactions
                if isinstance(node, Metabolite):
                    m = node
                else:
                    m = Metabolite()
                metabolites[node] = m
                # Add to compartments we constructed above.
                if 'compartment' in data:
                    c = data['compartment']
                    compartments[c].metabolites.add(m)

        for node, reaction in reactions_to_fix:
            # Get neighbours
            for neighbour, _ in graph.in_edges(node):
                # Get edge weight (i.e. stoichiometry)
                s = graph.get_edge_data(neighbour, node)['weight']
                # Add neighbours to reaction object.
                reaction.reactants[metabolites[neighbour]] += s
            for _, neighbour in graph.out_edges(node):
                # Get edge weight (i.e. stoichiometry)
                s = graph.get_edge_data(node, neighbour)['weight']
                # Add neighbours to reaction object.
                reaction.products[metabolites[neighbour]] += s

        for reaction, regulators in add_regulators:
            for m, regulation_type in regulators.items():
                try:
                    reaction.regulators[metabolites[m]] = regulation_type
                except KeyError:
                    raise ValueError("Mixing of Network objects and plain nodes not supported.")

        self.metabolites = set(metabolites.values())
        self.compartments = set(compartments.values())
        return self

    def to_sbml(self, level=(3, 1)):
        """Convert the Network to an SBML string."""
        # Check everything that returns int if it is libsbml.LIBSBML_OPERATION_SUCCESS.
        # if it is not raise a RuntimeError.
        # import decorate_library as dl
        # Also this is a horrible hack and should probably not be used.
        # decorator = dl.create_check_value_decorator(int, libsbml.LIBSBML_OPERATION_SUCCESS,
        #                                             RuntimeError)
        # with dl.decorate_library(libsbml, decorator):
        document = libsbml.SBMLDocument(*level)
        model = document.createModel()

        for compartment in self.compartments:
            c = model.createCompartment()
            c.setId(compartment.name)
            c.setConstant(True)
            c.setSize(compartment.volume)
            c.setSpatialDimensions(3)

        for metabolite in self.metabolites:
            s = model.createSpecies()
            s.setId(metabolite.name)
            s.setBoundaryCondition(False)
            s.setHasOnlySubstanceUnits(False)
            s.setConstant(False)
            for compartment in self.compartments:
                if metabolite in compartment.metabolites:
                    s.setCompartment(compartment.name)

        for reaction in self.reactions:
            r = model.createReaction()
            r.setId(reaction.name)
            r.setFast(False)
            r.setReversible(True)
            for reactant, s in reaction.reactants.items():
                rr = r.createReactant()
                rr.setSpecies(reactant.name)
                rr.setStoichiometry(-s)
                rr.setConstant(False)
            for product, s in reaction.products.items():
                rr = r.createProduct()
                rr.setSpecies(product.name)
                rr.setStoichiometry(s)
                rr.setConstant(False)
            for regulators, regulation_type in reaction.regulators.items():
                rr = r.createModifier()
                rr.setSpecies(regulators.name)
                rr.setSBOTerm(regulation_type)

        string = libsbml.writeSBMLToString(document)
        return string

    def to_networkx(self):
        """Convert the Network to a directed bipartite NetworkX graph.

        Reaction and metabolite objects are preserved as nodes.
        Edges are directed from reactant metabolites to reaction to product metabolites,

        Compartments are persevered as metabolite node properties.
        Regulators are reaction node properties.
        Edges are weighted by the stoichiometry of the reaction.

        Conversion back and from Network to NetworkX graph will preserve the structure, but
        might recreate objects.
        """
        G = nx.DiGraph()

        compartment_data = {}
        for compartment in self.compartments:
            compartment_data[compartment.name] = compartment.volume
        G.graph['compartments'] = compartment_data

        for metabolite in self.metabolites:
            if self.compartments:
                for compartment in self.compartments:
                    if metabolite in compartment.metabolites:
                        G.add_node(metabolite, bipartite=0, type='metabolite',
                                   compartment=compartment.name)
            else:
                G.add_node(metabolite, bipartite=0, type='metabolite')

        for reaction in self.reactions:
            G.add_node(reaction, bipartite=1, type='reaction')
            if reaction.regulators:
                nx.set_node_attributes(G, 'regulators', {reaction: reaction.regulators})
            for reactant, s in reaction.reactants.items():
                G.add_edge(reactant, reaction, weight=s)
            for product, s in reaction.products.items():
                G.add_edge(reaction, product, weight=s)

        return G

    def to_stoichiometry(self, extras=False):
        """Convert the Network to a representation of S, C and R."""
        n_metabolites, n_reactions = len(self.metabolites), len(self.reactions)
        n_compartments = len(self.compartments)
        S = np.zeros((n_metabolites, n_reactions), dtype=int)
        C = np.zeros(n_metabolites, dtype=int)
        R = np.zeros((n_metabolites, n_reactions), dtype=int)
        C_volumes = np.zeros(n_compartments)

        metabolites = sorted([i.name for i in self.metabolites])
        m_pos = dict(zip(sorted([i.name for i in self.metabolites]),
                         range(len(self.metabolites))))
        reactions = sorted([i.name for i in self.reactions])
        r_pos = dict(zip(sorted([i.name for i in self.reactions]),
                         range(len(self.reactions))))
        compartments = sorted([i.name for i in self.compartments])
        c_pos = dict(zip(sorted([i.name for i in self.compartments]),
                         range(len(self.compartments))))

        for reaction in self.reactions:
            for reactant, s in reaction.reactants.items():
                S[m_pos[reactant.name], r_pos[reaction.name]] = -s
            for product, s in reaction.products.items():
                S[m_pos[product.name], r_pos[reaction.name]] = s
            for regulator, regulation_type in reaction.regulators.items():
                R[m_pos[regulator.name], r_pos[reaction.name]] = regulation_type

        for compartment in self.compartments:
            for metabolite in self.metabolites:
                C[m_pos[metabolite.name]] = c_pos[compartment.name]
            C_volumes[c_pos[compartment.name]] = compartment.volume

        if not extras:
            return S, C, R
        else:
            return S, C, R, metabolites, m_pos, reactions, r_pos, compartments, c_pos, C_volumes

    def update_distributions(self, **kwargs):
        """Add parameter distributions for base quantities that can be used to draw models from.

        Required quantities: c, mu, kV, u, kM, kI, kA.
        """
        want = 'c', 'mu', 'kv', 'u', 'km', 'ki', 'ka', 'keq', 'kcat', 'kcat-', 'vmax', 'A', 'mu*'
        for i in want:
            if i not in kwargs and i not in self.distributions:
                raise ValueError("Missing {}".format(i))
        self.distributions.update(kwargs)

    def parameterize(self, random_seed=None):
        """Draw from the base distributions generating a full set of model parameters."""
        if random_seed is not None:
            np.random.seed(random_seed)

        want = 'c', 'mu', 'kv', 'u', 'km', 'ki', 'ka', 'keq', 'kcat', 'kcat-', 'vmax', 'A', 'mu*'
        for i in want:
            if i not in self.distributions:
                raise ValueError("Missing {}".format(i))
        parameters = []

        # Generate a value according to the distribution
        # Data format is (mean, sd, parameter type, compound, reaction)
        # If compound/reaction not available, use None.
        sd = 1e-12
        for metabolite in sorted(self.metabolites):
            for parameter_type in 'mu', 'c':
                mean = self.distributions[parameter_type].draw()
                parameters.append((mean, sd, parameter_type, metabolite.name, None))

        for reaction in sorted(self.reactions):
            for parameter_type in 'kv', 'u':
                mean = self.distributions[parameter_type].draw()
                parameters.append((mean, sd, parameter_type, None, reaction.name))

            for metabolite in sorted(reaction.reactants + reaction.products):
                parameter_type = 'km'
                mean = self.distributions[parameter_type].draw()
                parameters.append((mean, sd, parameter_type, metabolite.name, reaction.name))
            for metabolite, regulation_type in sorted(reaction.regulators.items()):
                if regulation_type in SBO_activators:
                    parameter_type = 'ka'
                elif regulation_type in SBO_inhibitors:
                    parameter_type = 'ki'
                mean = self.distributions[parameter_type].draw()
                parameters.append((mean, sd, parameter_type, metabolite.name, reaction.name))

        # Prepare for balancing
        data = parameters
        S, C, R, compounds, c_pos, reactions, r_pos, _, _, _ = self.to_stoichiometry(True)
        priors = {k: v.priors for k, v in self.distributions.items()}
        # Make sure they are balanced by only picking the base parameters and deriving the rest
        # without generating additional data.
        augment = False
        balancing_result = balancer.balance(compounds, reactions, S,
                                            data, priors, balancer.dependencies,
                                            balancer.nonlog, balancer.R, balancer.T,
                                            augment, c_pos, r_pos)

        # Save the complete set of mean values from the balancing and add
        # ki / ka since balancing ignores them.
        self.parameters = {}
        for (t, m, r), v in zip(balancing_result.columns, balancing_result.mean):
            self.parameters[(t, r, m)] = v, sd
        for (mean, sd, t, m, r) in parameters:
            if t in ('ka, ki'):
                self.parameters[(t, r, m)] = mean, sd

        return [k + v for k, v in self.parameters.items()]

    def generate_parameter_measurements(self, noise=.1, random_seed=None):
        """Using the generated parameter set and distributions to create noisy measurements.

        Noise is multiplicative noise drawn from a truncated random normal distribution
        limited by [0, 2].
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        noisy_parameters = dict(self.parameters)

        # Use a truncated normal distribution for noise.
        lower, upper = 0, 2
        mu, sigma = 1, noise
        a, b = (lower - mu) / sigma, (upper - mu) / sigma
        random_noise = scipy.stats.truncnorm(a, b, loc=mu, scale=noise).rvs(len(self.parameters))

        for i, key in enumerate(sorted(noisy_parameters.keys())):
            mu = random_noise[i] * noisy_parameters[key][0]
            sigma = abs(noise * mu)
            noisy_parameters[key] = (mu, sigma)

        return [k + v for k, v in noisy_parameters.items()]


class Compartment(object):
    C_UID = UID()

    def __init__(self, name=None, metabolites=None, volume=None):
        """Create an empty Compartment object."""
        self.name = name if name is not None else 'C_{}'.format(self.C_UID.get())
        self.metabolites = set() if metabolites is None else set(metabolites)
        self.volume = 1.0 if volume is None else volume

    def __repr__(self):
        return "Compartment(name='{}', metabolites={})".format(self.name, self.metabolites)

    def __str__(self):
        return self.name


class Metabolite(object):
    M_UID = UID()

    def __init__(self, name=None):
        """Create an empty Metabolite object."""
        self.name = name if name is not None else 'M_{}'.format(self.M_UID.get())

    def __repr__(self):
        return "Metabolite(name='{}')".format(self.name)

    def __str__(self):
        return self.name


class Reaction(object):
    R_UID = UID()

    def __init__(self, name=None, reactants=None, products=None, regulators=None):
        """Create an empty Reaction object."""
        self.name = name if name is not None else 'R_{}'.format(self.R_UID.get())
        counter = collections.Counter
        self.reactants = counter() if reactants is None else counter(reactants)
        self.products = counter() if products is None else counter(products)
        self.regulators = {} if regulators is None else regulators

    def __repr__(self):
        if self.regulators:
            regulators = ', regulators={}'.format(self.regulators)
        else:
            regulators = ''
        return "Reaction(name='{}', reactants={}, products={}{})".format(self.name,
                                                                         self.reactants,
                                                                         self.products,
                                                                         regulators)

    def __str__(self):
        return self.name


class Distribution(object):
    def __init__(self):
        raise NotImplementedError("Should be overwritten by child class.")

    def draw(self):
        raise NotImplementedError("Should be overwritten by child class.")

    def priors(self):
        raise NotImplementedError("Should be overwritten by child class.")


class NormalDistribution(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def draw(self, Z=None):
        if Z is None:
            Z = np.random.normal(0, 1)
        return self.mu + self.sigma * Z

    @property
    def priors(self):
        return self.mu, self.sigma

    def __repr__(self):
        return "x ~ N({}, {}**2)".format(self.mu, self.sigma)


class LogNormalDistribution(Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    @classmethod
    def from_linear(cls, mean=None, median=None, sd=None):
        if mean is not None and sd is not None and median is None:
            v = np.square(sd)
            mu = np.log(mean / (np.sqrt(1 + v / np.square(mean))))
            sigma = np.sqrt(np.log(1 + v / np.square(mean)))
            return cls(mu, sigma)
        elif median is not None and sd is not None and mean is None:
            return cls(np.log(median), np.log(sd))
        else:
            raise ValueError('Requires mean xor median and sd')

    def draw(self, Z=None):
        if Z is None:
            Z = np.random.normal(0, 1)
        return np.exp(self.mu + self.sigma * Z)

    @property
    def priors(self):
        return self.mu, self.sigma

    def __repr__(self):
        return "ln(x) ~ N({}, {}**2)".format(self.mu, self.sigma)


def generate_stoichiometry(n_metabolites, n_reactions, n_compartments, n_regulators,
                           max_degree, gamma, regulatory_types, ensure_no_empty_reactions=True,
                           random_seed=None):
    """Very simple function to generate some random networks.

    Note that this suffers from several problems:
        Unreachable nodes
        Empty reactions
        No limit on only input/only output reactions

    The empty reactions can be prevented by looping until no empty reactions are found.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Create power law distribution up to max degree
    p = np.power(np.arange(1.0, max_degree), -gamma)
    # Normalize to 1
    p = (p / p.sum()).cumsum()

    while True:
        # Find insertion indices into the cumulative distribution, this corresponds to the degree.
        metabolite_degree = np.searchsorted(p, np.random.rand(n_metabolites)) + 1

        # Divide metabolites over reactions
        S = np.zeros((n_metabolites, n_reactions), dtype=int)
        for i, d in enumerate(metabolite_degree):
            choice = np.random.choice(n_reactions, d)
            for j in choice:
                S[i, j] += 1

        # Divide into products and reactants
        for i in range(n_reactions):
            indices = S[:, i] != 0
            length = indices.sum()
            if length == 0:
                continue
            signs = np.zeros(length, dtype=int)
            # In/or output?
            if length >= 1:
                signs[0] = np.random.choice((-1, 1), 1)
            # More then one, add the one we didn't add yet
            if length >= 2:
                signs[1] = -signs[0]
            # Assign the rest randomly.
            if length > 2:
                signs[2:] = np.random.choice((-1, 1), length - 2)
            S[indices, i] *= signs

        empties = np.where(np.sum(np.abs(S), axis=0) == 0)[0]

        if not ensure_no_empty_reactions:
            if empties.size > 0:
                warnings.warn("Empty reaction columns: {}".format(empties))
            break
        elif not empties.size:
            break

    in_degree = (S < 0).sum(axis=0)
    out_degree = (S > 0).sum(axis=0)
    high_degree = np.where(np.logical_or(in_degree > 3, out_degree > 3))[0]
    if high_degree.size > 3:
        warnings.warn("High in/out degree (>3) for reactions {}".format())

    # Randomly assign each metabolite to a compartment
    C = np.random.choice(n_compartments, n_metabolites)

    # Randomly assign metabolites to a reaction as a regulator.
    # Choose reaction, metabolites and types.
    reactions = np.random.choice(n_reactions, n_regulators)
    metabolites = np.random.choice(n_metabolites, n_regulators)
    # Positive is activation, negative is inhibition.
    types = np.random.choice(regulatory_types, n_regulators)

    R = np.zeros(S.shape, dtype=int)
    for m, r, t in zip(metabolites, reactions, types):
        R[m, r] = t

    return S, C, R


if __name__ == '__main__':
    n_metabolites = 10
    n_reactions = 8
    n_compartments = 2
    n_regulators = 0
    max_degree = 8
    gamma = 1.5
    regulatory_types = [206, 537, 636, 533]
    S, C, R = generate_stoichiometry(n_metabolites, n_reactions, n_compartments, n_regulators,
                                     max_degree, gamma, regulatory_types)
    N = Network.from_stoichiometry(S, C, R)
    G = N.to_networkx()

    # This actually duplicates nodes? Bug? Probably has something to do with the comparison
    # of Reaction/Metabolite objects
    connected = nx.is_connected(nx.convert_node_labels_to_integers(G.to_undirected()))
    # print("Graph is connected: {}".format(connected))

    # If the graph is not connected, there might be multiple feasible bipartite sets.
    # Also, this doesn't actually use the bipartite attribute of the nodes?
    rs, ms = nx.bipartite.sets(G)
    assert(rs == set(N.reactions) or not connected)
    assert(ms == set(N.metabolites) or not connected)

    print("Metabolite degrees: {}".format(sorted(G.degree(N.metabolites).values())))
    print("Reaction degrees: {}".format(sorted(G.degree(set(N.reactions)).values())))

    # We need to make sure the sorting order is constant if we want to check the stoichiometry
    sort_ = lambda xs: sorted(xs, key=lambda x: int(x.name.split('_')[1]))
    s1 = nx.bipartite.biadjacency_matrix(G, sort_(N.metabolites), sort_(N.reactions))
    s2 = nx.bipartite.biadjacency_matrix(G, sort_(N.reactions), sort_(N.metabolites))
    print("Stoichiometry matches: {}".format(np.all(-s1 + s2.T == S)))

    # import matplotlib.pyplot as plt
    # plt.ion()
    # colors = ''.join(['b' if i[1]['bipartite'] else 'g' for i in G.nodes(data=True)])
    # nx.draw_networkx(G, node_color=colors)
    sbml = N.to_sbml()

    # Check loop from networkx to Network and back.
    N2 = Network.from_networkx_graph(G)
    G2 = N2.to_networkx()

    # Compare graphs, checking shape and edge lengths and compartment data.
    import networkx.algorithms.isomorphism as iso
    matcher = iso.numerical_edge_match('weight', 1.0)
    assert(nx.is_isomorphic(G, G2, edge_match=matcher))
    assert(sorted(G.nodes()) == sorted(G2.nodes()))
    assert(G.graph == G2.graph)

    # Check loop from networkx to Network and back, but with a stripped down version.
    # This will not work if regulators are added, since the node and the referenced regulator
    # won't match up after stripping the network nodes down.
    if not n_regulators:
        N3 = Network.from_networkx_graph(nx.convert_node_labels_to_integers(G))
        G3 = N3.to_networkx()
        assert(nx.is_isomorphic(G, G3, edge_match=matcher))
        assert(nx.is_isomorphic(G2, G3, edge_match=matcher))
    assert(G.graph == G3.graph)

    lognormal = 'c', 'kv', 'u', 'km', 'ki', 'ka', 'keq', 'kcat', 'kcat-', 'vmax'
    normal = 'mu', 'mu*', 'A'
    distributions = {}
    for i in lognormal:
        distributions[i] = LogNormalDistribution(0., 1.)
    for i in normal:
        distributions[i] = NormalDistribution(0., 1.)

    N.update_distributions(**distributions)
    parameters = N.parameterize()

    possible_measurements = N.generate_parameter_measurements(0.1)

    # Check that the network structure doesn't change if we use the same
    # seed with or without regulation and compartments.
    n_metabolites = 10
    n_reactions = 8
    max_degree = 8
    gamma = 1.5
    seed = 1

    regulatory_types = [[20, 462], [206, 537, 636, 533]]
    n_compartments = [1, 3]
    n_regulators = [1, 3]

    all_S = []
    import itertools
    for rt, nc, nr in itertools.product(regulatory_types, n_compartments, n_regulators):
        S, C, R = generate_stoichiometry(n_metabolites, n_reactions, nc, nr,
                                         max_degree, gamma, rt,
                                         ensure_no_empty_reactions=True,
                                         random_seed=seed)
        all_S.append(S)

    # Compare all to first result.
    all_same_S = np.all([np.all(all_S[0] == all_S[i]) for i in range(len(all_S))])
    assert(all_same_S)
