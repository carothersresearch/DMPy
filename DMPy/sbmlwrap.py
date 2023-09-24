#!/usr/bin/env python
"""
Convenience wrapper for the SBML module.

Includes automatic extraction of identifiers, parameters and kinetics, as well
as several convenience functions and conversion of SBML/MathML to Sympy objects.

This is mostly a read only wrapper - only model.kineticize will update the
reactions and local parameters of the underlying SBML model.

Author: Rik van Rosmalen
"""
from __future__ import division
from __future__ import print_function

# Standard library
import collections
import logging
import os
import re
import operator

# Scientific libraries
import numpy as np
import sympy

import libsbml

# Project imports
from identifiers import Identifiers

try:
    import sympy.printing.llvmjitcode as jit
except ImportError:
    logging.info("SBML:ODE: Could not load experimental Sympy LLVM JIT backend.")
    jit = False

# Modify MathML printing for LibSBML compatibility.
from sympy.printing.mathml import MathMLPrinter

def _print_Symbol(self, sym):
    """Modified version of sympy.printing.mathml.MathMLPrinter._print_Symbol.

    This removes the replacement of greek characters, and the conversion of
    _ and ^ to sub and superscript, which will allow it to do a round trip
    through sympy <-> libsbml.
    """
    ci = self.dom.createElement(self.mathml_tag(sym))
    ci.appendChild(self.dom.createTextNode(sym.name))
    return ci

def _print_float(self, e):
    """For some reason the printer only has int defined and not float so we add it ourselves."""
    x = self.dom.createElement(self.mathml_tag(e))
    x.appendChild(self.dom.createTextNode(str(e)))
    return x

def mathml_tag(self, e):
    """Modified version of sympy.printing.mathml.MathMLPrinter.mathml_tag

    Adds the mapping of float -> mathml: cn

    Returns the MathML tag for an expression."""
    translate = {
        'Add': 'plus',
        'Mul': 'times',
        'Derivative': 'diff',
        'Number': 'cn',
        'int': 'cn',
        'float': 'cn',
        'Pow': 'power',
        'Symbol': 'ci',
        'Integral': 'int',
        'Sum': 'sum',
        'sin': 'sin',
        'cos': 'cos',
        'tan': 'tan',
        'cot': 'cot',
        'asin': 'arcsin',
        'asinh': 'arcsinh',
        'acos': 'arccos',
        'acosh': 'arccosh',
        'atan': 'arctan',
        'atanh': 'arctanh',
        'acot': 'arccot',
        'atan2': 'arctan',
        'log': 'ln',
        'Equality': 'eq',
        'Unequality': 'neq',
        'GreaterThan': 'geq',
        'LessThan': 'leq',
        'StrictGreaterThan': 'gt',
        'StrictLessThan': 'lt',
    }

    for cls in e.__class__.__mro__:
        n = cls.__name__
        if n in translate:
            return translate[n]
    # Not found in the MRO set
    n = e.__class__.__name__
    return n.lower()

# Override the sympy definition.
MathMLPrinter._print_Symbol = _print_Symbol
MathMLPrinter._print_float = _print_float
MathMLPrinter.mathml_tag = mathml_tag

# -----------------------------------------------------------------------------
#                              Custom exceptions
# -----------------------------------------------------------------------------


class FileNotFoundError(OSError):
    """Exception for use in python 2.7."""

    pass


class SBMLError(Exception):
    """Exception for errors from within libsbml."""

    pass


# -----------------------------------------------------------------------------
#                                 Classes
# -----------------------------------------------------------------------------


class Model(object):
    """Wrapper for the SBML model object from libsbml.

    Used to handle error checking and return appropriate SBMLError exceptions
    instead.
    """

    def __init__(self, path=None):
        """Create the sbml_model from the file at path.

        Arguments:
        path -- file path string, either relative to current directory or absolute.
        """
        if path is not None:
            self.path = path
            self._load_model(path)
            self._updateReferences()
            logging.info("SBML: Loaded model from {} with no errors.".format(path))
        else:
            self.path = ""

    def __str__(self):
        """Short description."""
        return "SBML Model({})".format(self.name)

    def __repr__(self):
        """Short description."""
        return str(self)

    @classmethod
    def from_string(cls, string):
        """Initialize a model from a string instead of a file."""
        M = cls(None)
        M.name = ""
        M.path = ""

        # Load the SBML (Covert to ascii as libsbml doesn't like unicode.)
        string = string.encode('ascii', 'xmlcharrefreplace')
        M.document = libsbml.readSBMLFromString(string)

        # Check for document errors and print if exists.
        if M.document.getNumErrors() > 0:
            M.document.printErrors()
            raise SBMLError("Errors found in model document!")

        M.model = M.document.getModel()
        M.sbml_object = M.model
        M._updateReferences()

        logging.info("SBML: Loaded model from string with no errors.")

        return M

    def reload(self):
        """Reload the model from the same path as previously in case the file is changed."""
        if self.path():
            return self.__init__(self.path)
        else:
            raise FileNotFoundError("Cannot reload model because no path is know for the model.")

    def _load_model(self, path):
        """Load the SBML model at path.

        The file can be compressed (.gz, .bz2, .zip) if the version of libsbml supports it.

        Arguments:
        path -- file path string, either relative to current directory or absolute.
        """
        # Since the libsbml python bindings are very thin, we need to be careful with errors.
        if not os.path.exists(path):
            raise FileNotFoundError("File does not exist: {}".format(path))

        reader = libsbml.SBMLReader()

        # Check for decompression libraries if required. # TODO: Use python extraction libs.
        if path.endswith('.zip') or path.endswith('.gz'):
            if not reader.hasZlib():
                raise SBMLError("Zlib (.zip/.gz) not supported in current libsbml version.")
        elif path.endswith('.bz2'):
            if not reader.hasBzip2():
                raise SBMLError("Bzip2 (.bz2) not supported in current libsbml version.")

        self.document = reader.readSBML(path)

        # Check for document errors and print if exists.
        if self.document.getNumErrors() > 0:
            self.document.printErrors()
            raise SBMLError("Errors found in model document: {}".format(path))

        self.model = self.document.getModel()
        self.sbml_object = self.model

        try:
            self.name = self.model.getName()
        except (ValueError, AttributeError):
            self.name = ""

    def _updateReferences(self):
        """Update the internal references of species and reaction objects."""
        # First load everything.
        compartments = (Compartment(self.model.getCompartment(i))
                        for i in range(self.model.getNumCompartments()))
        species = (Species(self.model.getSpecies(i))
                   for i in range(self.model.getNumSpecies()))
        reactions = (Reaction(self.model.getReaction(i))
                     for i in range(self.model.getNumReactions()))

        # Now create lookup by id.
        self.compartments = collections.OrderedDict(((i.id, i) for i in compartments))
        self.species = collections.OrderedDict(((i.id, i) for i in species))
        self.reactions = collections.OrderedDict(((i.id, i) for i in reactions))

        # Species compartments
        for id_, species in self.species.items():
            species.compartment = self.compartments[species.compartment]
        # Reaction compartments and products/reactants/modifiers.
        for id_, reaction in self.reactions.items():
            reaction.species = {self.species[name] for name in reaction.species}
            reaction.reactants = [(self.species[name], i) for name, i in reaction.reactants]
            reaction.products = [(self.species[name], i) for name, i in reaction.products]
            reaction.modifiers = [self.species[name] for name in reaction.modifiers]

            if reaction.compartment is not None and reaction.compartment:
                reaction.compartment = self.compartments[reaction.compartment]
            else:  # Compartment not set. Can we get it from the reactants/products?
                compartments = collections.Counter()
                for species, i in reaction.reactants + reaction.products:
                    compartments[species.compartment]
                for species in reaction.modifiers:
                    compartments[species.compartment]
                reaction.compartment = compartments.most_common(1)

        # Global parameters
        self.parameters = collections.OrderedDict()
        for i in range(self.model.getNumParameters()):
            p = self.model.getParameter(i)
            self.parameters[p.getId()] = p.getValue()
        # Compartment sizes can be parameters.
        for compartment in self.compartments.values():
            self.parameters[compartment.id] = compartment.sbml_object.getSize()

    def get_stoichiometry_matrix(self):
        """Get the stoichiometry matrix for the model."""
        stochiometry = np.zeros((len(self.species), len(self.reactions)))

        s_pos = dict(((j, i) for i, j in enumerate(self.species.keys())))

        for r_pos, reaction in enumerate(self.reactions.values()):
            # Reactants should have a negative term instead.
            reactants = [(i, -j) for i, j in reaction.reactants]
            for compound, coefficient in reaction.products + reactants:
                stochiometry[s_pos[compound.id], r_pos] = coefficient

        return stochiometry

    def get_state(self, array=False):
        """Get the initial concentration state of the model."""
        if not array:
            initial_state = collections.OrderedDict()

            for id_, species in self.species.items():
                initial_state[id_] = species.sbml_object.getInitialConcentration()
        else:
            initial_state = np.zeros(len(self.species))

            s_pos = dict(((j, i) for i, j in enumerate(self.species.keys())))
            for compound in self.species.values():
                initial_state[s_pos[compound.id]] = compound.sbml_object.getInitialConcentration()

        return initial_state

    def run_initial_assignments(self):
        """Calculate any initial assignments and updated the referenced value."""
        parameters = self.get_state()
        parameters.update(self.parameters)

        for i in range(self.model.getNumInitialAssignments()):
            assignment = self.model.getInitialAssignment(i)
            # Convert to Sympy, substitute and calculate value
            target = assignment.getSymbol()
            value = ASTNodeWrapper(assignment.getMath()).to_sympy().subs(parameters)
            logging.info("SBML: Initial assignments: {} changed to {}.".format(target, value))

            # Check where it needs to go.
            if target in self.compartments:
                # Set volume for compartment.
                sbml_obj = self.compartments[target].sbml_object
                sbml_obj.setSize(float(value))
                self.parameters[target] = float(value)
            elif target in self.parameters:
                # Set value for parameters
                for i in range(self.model.getNumParameters()):
                    sbml_obj = self.model.getParameter(i)
                    if sbml_obj.getId() == target:
                        break
                sbml_obj.setValue(float(value))
                self.parameters[target] = float(value)
            elif target in self.species:
                # Set initial quantity for species
                sbml_obj = self.species[target].sbml_object
                sbml_obj.setInitialConcentration(float(value))
            else:
                raise ValueError("Invalid target {} for initial assignment {}".
                                 format(target, assignment.getId()))

    def inline_rules(self, ruletypes=('assignment')):
        """Place any rules defining functions into the kinetic rate equation itself."""
        # According to the specs, rules must hold at all times which means they can be inlined,
        # although it might be more efficient to group it up and avoid repeated calculation of
        # each rule defined variable, doing so requires a better code generation step.
        rules = collections.OrderedDict()
        for i in range(self.model.getNumRules()):
            rule = self.model.getRule(i)
            inline = (('rate' in ruletypes and rule.isRate()) or
                      ('assignment' in ruletypes and rule.isAssignment()) or
                      ('algebraic' in ruletypes and rule.isAlgebraic()) or
                      ('compartment_volume' in ruletypes and rule.isCompartmentVolume()) or
                      ('parameter' in ruletypes and rule.isParameter()) or
                      ('scalar' in ruletypes and rule.isScalar()) or
                      ('species_concentration' in ruletypes and rule.isSpeciesConcentration()))
            if inline and rule.isSetMath():
                logging.info("SBML:ODE: Marked rule for {} for inlining.".format(rule.variable))
                rules[rule.variable] = ASTNodeWrapper(rule.getMath()).to_sympy()
            else:
                raise ValueError('Rule not inlined. {}'.format(rule.id))

        # Create dependency graph
        dependencies = collections.defaultdict(set)
        for rule_id in rules.keys():
            for other_rule_id, other_rule_expr in rules.items():
                if sympy.Symbol(rule_id) in other_rule_expr.free_symbols:
                    dependencies[other_rule_id].add(rule_id)

        # Resolve dependency graph order (i.e. topological sort)
        keys = set(rules.keys())
        order = []
        resolved = set()

        while keys - resolved:
            # Everything in a single layer is only dependent on the layers above.
            layer = set()
            # Check if we resolved everything already for this element?
            for key in keys - resolved:
                if dependencies[key] <= resolved:
                    layer.add(key)
            # Updated resolved keys with the new layer
            resolved |= layer
            # If you want to see the layers use append
            # order.append(layer)
            # We don't care so go for flat
            order.extend(layer)

            # If nothing gets resolved in a single loop, this means we're stuck on a cycle.
            if not layer and (keys - resolved):
                print(dependencies)
                raise ValueError('Cyclic dependency graph in rules')

        # Loop trough reactions kinetics
        for id_, reaction in self.reactions.items():
            expr = ASTNodeWrapper(reaction.sbml_object.getKineticLaw().getMath()).to_sympy()
            # Try all substitutions in reverse dependent order.
            for rule_id in order[::-1]:
                rule_expr = rules[rule_id]
                expr = expr.subs(rule_id, rule_expr)
            # Convert the new Sympy expression back to a ASTNode and update model.
            ast = ASTNodeWrapper.from_sympy(expr).node
            exit = reaction.sbml_object.getKineticLaw().setMath(ast)
            if exit == libsbml.LIBSBML_INVALID_OBJECT:
                raise ValueError('kineticLaw.setMath() invalid input: {}.'.format(expr))

    def inline_functions(self):
        """Place any functions in the kinetics into the kinetic rate equation itself."""
        # Parse functions
        functions = {}
        for i in range(self.model.getNumFunctionDefinitions()):
            function = self.model.getFunctionDefinition(i)
            functions[function.getId()] = ASTNodeWrapper(function.getMath()).to_sympy()

        for id_, reaction in self.reactions.items():
            expr = ASTNodeWrapper(reaction.sbml_object.getKineticLaw().getMath()).to_sympy()
            # Walk the tree
            for sub_expr in sympy.preorder_traversal(expr):
                # Check for functions
                if hasattr(sub_expr, 'is_Function') and sub_expr.is_Function:
                    # Check the name
                    name = sub_expr.func.__name__
                    if name in functions:
                        # Replace if know
                        expr = expr.replace(sympy.Function(name), functions[name])
                    # Disabled because this would be raised as well on Sympy built-in functions
                    # such as log / sin. Not sure how to differentiate between custom and
                    # built-in Sympy functions.
                    # else:
                        # raise ValueError("Unknown function: {} cannot be inlined.".format(name))

            # Convert the new Sympy expression back to a ASTNode and update model.
            ast = ASTNodeWrapper.from_sympy(expr).node
            exit = reaction.sbml_object.getKineticLaw().setMath(ast)
            if exit == libsbml.LIBSBML_INVALID_OBJECT:
                raise ValueError('kineticLaw.setMath(ast) invalid input.')

    def get_formulas(self, species_or_reactions='both', parameters=False,
                     enforce_reversibility=True, compensate_volumes=True):
        """Create the model reaction and species formulas for use in further analysis."""
        reactions = collections.OrderedDict()
        species = collections.OrderedDict()
        # Fill this dictionary up already in case some species are not reactants and thus
        # do not change over time.
        for compound in self.species.values():
            species[compound] = 0

        for id_, reaction in self.reactions.items():
            sbml_obj = reaction.sbml_object
            reactions[id_] = ASTNodeWrapper(sbml_obj.getKineticLaw().getMath()).to_sympy()
            if (enforce_reversibility and
                    sbml_obj.isSetReversible() and not
                    sbml_obj.getReversible()):
                reactions[id_] = sympy.Max(reactions[id_], 0)
                logging.info("SBML:ODE: {} set as irreversible.".format(reaction.name))
            if parameters:
                parameters = self.parameters.copy()
                parameters.update({k: float(p['value']) for k, p in reaction.parameters.items()})
                if hasattr(reactions[id_], 'subs'):
                    reactions[id_] = reactions[id_].subs(parameters)

            if species_or_reactions in ('both', 'species'):
                reaction_compounds = reaction.products + [(i, -j) for i, j in reaction.reactants]

                for compound, coefficient in reaction_compounds:
                    # Calculate conversion vs. reference.
                    if compensate_volumes:
                        compartment = compound.compartment.size
                        species[compound] += coefficient * reactions[id_] * 1 / compartment
                    else:
                        species[compound] += coefficient * reactions[id_]

        if species_or_reactions == 'species':
            return species
        elif species_or_reactions == 'reactions':
            return reactions
        else:
            return species, reactions

    def get_species_formulas(self, parameters=False, enforce_reversibility=True,
                             compensate_volumes=True):
        """Get the reaction ODE formula based on the species."""
        return self.get_formulas('species', parameters, enforce_reversibility, compensate_volumes)

    def get_reaction_formulas(self, parameters=False, enforce_reversibility=True):
        """Get the reaction ODE formula based on the reaction."""
        return self.get_formulas('reactions', parameters, enforce_reversibility,
                                 False)

    def get_ode_function(self, disabled_metabolite_fluxes=None,
                         enforce_reversibility=True, compensate_volumes=True,
                         jacobian=False):
        """Create a function for use by a numerical integrator such as scipy.integrate.ode."""
        if disabled_metabolite_fluxes is None:
            disabled_metabolite_fluxes = set()
        # Start with getting the model into shape.
        self.run_initial_assignments()
        self.inline_functions()
        self.inline_rules()

        species_idx = {name: idx for idx, name in enumerate(self.species.keys())}

        boundary_condition = set() | set(disabled_metabolite_fluxes)
        for idx, species in enumerate(self.species.values()):
            if (species.sbml_object.isSetBoundaryCondition and
                    species.sbml_object.getBoundaryCondition()):
                boundary_condition.add(idx)
                logging.info("SBML:ODE: {} set as boundary metabolite.".format(species.name))

        # Alternatively, we can make a reaction centric view and take dot product with the
        # Stoichiometry (and make sure to account for compartment volumes!)
        S = self.get_stoichiometry_matrix()
        if compensate_volumes:  # Note that this is fixed so it won't work with changing volumes.
            # Check if compartment sizes are fixes, else error out.
            for c in self.compartments.values():
                if not c.sbml_object.constant:
                    raise NotImplementedError("Variable compartment sizes not yet implemented!")
            T = np.array([i.compartment.size for i in self.species.values()])
            S = (S.T * 1/T).T
        boundary_condition_array = np.array(list(boundary_condition))

        species, reactions = self.get_formulas('both', True, enforce_reversibility,
                                               compensate_volumes)

        if not jit or enforce_reversibility:
            reaction_functions = []
            for reaction_formula_id, formula in reactions.items():
                if formula:
                    args = list(formula.free_symbols)
                    f = sympy.lambdify(args, formula)
                    reaction_functions.append((f, [species_idx[arg.name] for arg in args]))
                else:
                    reaction_functions.append((False, False))

            def func(t, state):
                reaction_flux = np.zeros(S.shape[1])
                for i, (f, args) in enumerate(reaction_functions):
                    if not f:
                        pass
                    elif args:
                        reaction_flux[i] = f(*state[args])
                    else:
                        reaction_flux[i] = f()

                # Requires an transition matrix to compensate for different volumes.
                flux = S.dot(reaction_flux)
                if boundary_condition_array.shape[0] > 0:
                    flux[np.array(boundary_condition_array)] = 0
                return flux

        else:
            print("Using experimental LLVM function generation...")
            args = sympy.symbols(', '.join(list(self.species.keys())))
            expressions = list(reactions.values())
            expressions = [sympy.factor_terms(i) for i in expressions]
            # Simplify and eliminate common sub-expressions
            simplified = sympy.cse(expressions)
            # Create LLVM function.
            jit_f = jit.llvm_callable(args, simplified)

            def func(t, state):
                flux = S.dot(np.array(jit_f(*state)))
                if boundary_condition_array.shape[0] > 0:
                    flux[np.array(boundary_condition_array)] = 0
                return flux

        if not jacobian:
            return func

        jac_functions = []
        for species_formula_id, formula in species.items():
            for idx, species_diff in enumerate(self.species):
                f = sympy.diff(formula, sympy.Symbol(species_diff))
                if f:
                    args = list(f.free_symbols)
                    jac_functions.append((f, args))
                else:
                    jac_functions.append((f, False))

        if not jit or enforce_reversibility:
            def heaviside(x):
                if x > 0:
                    return 1
                elif x < 0:
                    return 0
                return .5

            modules = [{'Heaviside': heaviside},  "math", "mpmath", "numpy", "sympy"]

            jac_functions_lambdified = []
            for formula, args in jac_functions:
                if not formula:
                    jac_functions_lambdified.append((formula, args))
                else:
                    f = sympy.lambdify(args, formula, modules=modules)
                    args = [species_idx[arg.name] for arg in args]
                    jac_functions_lambdified.append((f, args))

            def jac(t, state):
                """Dynamically generated Jacobian function."""
                # Start with flat array
                jac = np.zeros(len(state)**2)
                for i, (f, args) in enumerate(jac_functions_lambdified):
                    # If 0, pass
                    if not f or i in boundary_condition:
                        pass
                    elif args:
                        jac[i] = f(*state[args])
                    else:
                        jac[i] = f()

                # Reshape
                return jac.reshape(len(state), len(state))
        else:
            args = sympy.symbols(', '.join(list(self.species.keys())))
            expressions = [i[0] for i in jac_functions]
            expressions = [sympy.factor_terms(i) for i in expressions]
            # Simplify and eliminate common sub-expressions
            simplified = sympy.cse(expressions)
            # Create LLVM function.
            jit_f_jac = jit.llvm_callable(args, simplified)

            def jac(t, state):
                return np.array(jit_f_jac(*state)).reshape(len(state), len(state))

        return func, jac

    def kineticize(self, parameters_path, regulation=None, rate_law='CM', version='cat',
                   cooperativities=None, ignore_reactions=None, T=None,
                   ignore_concentrations=False):
        """kineticize the model with common modular rate laws using the provided parameters.

        Arguments:
        parameters_path -- file location for the tab separated value parameter file.
        rate_law -- Type of rate law (CM / SM / DM / FM / PM) or dictionary of {reaction: rate law}
        version -- Numerator version (cat / hal / weg)
        regulation -- dictionary of {reaction: [
                                                (compound,
                                                 [inhibition/activation constants],
                                                 'activation' or 'inhibition',
                                                 'specific' or 'partial' or 'complete' or 'simple',
                                                 (...)
                                                 ]}
        cooperativities -- dictionary of {reaction: cooperativity factor}
        ignore_reactions -- Keep these reactions as defined in the SBML file.
        T -- temperature for use in weg type rate law

        For the details and mathematics behind these rate laws see:
        Liebermeister, W., Uhlendorf, J., & Klipp, E. (2010). Modular rate laws for enzymatic
        reactions: thermodynamics, elasticities and implementation. Bioinformatics, 26(12),
        1528-1534. http://doi.org/10.1093/bioinformatics/btq141
        """
        if regulation is None:
            regulation = {}
        if ignore_reactions is None:
            ignore_reactions = []
        if cooperativities is None:
            cooperativities = {}
        cooperativities = collections.defaultdict(lambda: 1, cooperativities)
        # If only a single rate law is provided, make it into a default dictionary.
        # else we can use the provided dictionary.
        if isinstance(rate_law, str):
            rate_laws = collections.defaultdict(lambda: rate_law)
        else:
            rate_laws = rate_law

        # Check which parameters we should construct.
        required_parameters = {'cat': ('u', 'km', 'c', "c'", 'kcat'),
                               'hal': ('u', 'km', 'c', "c'", 'kV', 'keq'),
                               'weg': ('u', 'km', 'c', "c'", 'kV', 'mu0', 'R', 'T')}[version]

        if version == 'weg' and T is None:
            raise ValueError("Temperature needs to be explicitly provided for Wegscheider "
                             " version of the common modular rate law")

        # Parse parameters
        parameter_values = {}
        with open(parameters_path, 'r') as f:
            # Skip header
            f.readline()
            for line in f:
                if not line.strip():
                    continue
                # 0-QuantityType, 1-SBMLReactionID, 2-SBMLSpeciesID, 3-Value, 4-Mean, 5-Std,
                # 6-Unit, 7-Provenance, 8-Type, 9-Source, 10-logMean, 11-logStd, 12-Temperature,
                # 13-pH, 14-Minimum, 15-Maximum
                values = [i.strip() for i in line.strip().split('\t')]
                parameter_values[(values[0], values[1], values[2])] = values[3]

        # Update initial concentrations if given.
        if not ignore_concentrations:
            for species in self.species.values():
                id_ = ('concentration', '', species.id)
                if id_ in parameter_values:
                    species.sbml_object.setInitialConcentration(float(parameter_values[id_]))

        # Save results to return for possible inspection
        reactions = {}

        # Create global symbols as required.
        if 'c' in required_parameters:
            con = {}
            for term in self.species.values():
                con[term.id] = sympy.symbols(term.id)

        if 'R' in required_parameters:
                R = 8.3144598

        for name, reaction in self.reactions.items():
            # Skip ignored reactions
            if name in ignore_reactions:
                continue

            # Retrieve kinetic law or create one.
            kinetic_law = reaction.sbml_object.getKineticLaw()
            if kinetic_law is None:
                kinetic_law = reaction.sbml_object.createKineticLaw()
            # Remove old parameters
            while kinetic_law.getNumParameters():
                kinetic_law.removeParameter(0)
            while kinetic_law.getNumLocalParameters():
                kinetic_law.removeLocalParameter(0)

            sbml_parameters = []

            # Create local symbols and helper variables as required.
            h = cooperativities[reaction.id]
            reactants = reaction.reactants
            products = reaction.products
            species = [(i, -s) for i, s in reactants] + products  # Note the `-`!
            rate_law = rate_laws[reaction.id]

            if 'u' in required_parameters:
                id_ = 'u_{}'.format(reaction.id)
                sbml_parameters.append((id_,
                                        'concentration of enzyme',
                                        reaction.id))
                u = sympy.symbols(id_)

            if 'kcat' in required_parameters:
                id_ = 'kcatsub_{}'.format(reaction.id)
                sbml_parameters.append((id_,
                                        'substrate catalytic rate constant',
                                        reaction.id))
                kf = sympy.symbols(id_)
                id_ = 'kcatprod_{}'.format(reaction.id)
                sbml_parameters.append((id_,
                                        'product catalytic rate constant',
                                        reaction.id))
                kr = sympy.symbols(id_)

            if 'km' in required_parameters:
                kms = {}
                for term, s in reaction.reactants + reaction.products:
                    id_ = 'kM_{}_{}'.format(reaction.id, term.id)
                    sbml_parameters.append((id_,
                                            'Michaelis constant',
                                            term.id))
                    kms[term.id] = sympy.symbols(id_)

            if "c'" in required_parameters:
                relcon = {}
                for term, s in reaction.reactants + reaction.products:
                    relcon[term.id] = con[term.id] / kms[term.id]

            if 'kV' in required_parameters:
                id_ = 'kV_{}'.format(reaction.id)
                sbml_parameters.append((id_,
                                        'catalytic rate constant geometric mean',
                                        reaction.id))
                kV = sympy.symbols(id_)

            if 'keq' in required_parameters:
                id_ = 'keq_{}'.format(reaction.id)
                sbml_parameters.append((id_,
                                        'equilibrium constant',
                                        reaction.id))
                keq = sympy.symbols(id_)

            if 'mu0' in required_parameters:
                mus = {}
                for term, s in reaction.reactants + reaction.products:
                    id_ = 'mu0_{}_{}'.format(reaction.id, term.id)
                    sbml_parameters.append((id_,
                                            'standard chemical potential',
                                            term.id))
                    mus[term.id] = sympy.symbols(id_)

            # Add new parameters to SBML
            for pname, ptype, id_ in sbml_parameters:
                p = kinetic_law.createParameter()
                p.initDefaults()
                p.setId(pname)
                p.setName(pname)
                # Compound and Reaction based property
                if ptype == 'Michaelis constant':
                    id_ = (ptype, reaction.id, id_)
                # Compound based property only.
                elif ptype == 'standard chemical potential':
                    id_ = (ptype, '', id_)
                # Reaction based property
                else:
                    id_ = (ptype, id_, '')
                p.setValue(float(parameter_values[id_]))

            # Construct numerator
            if version == 'cat':
                numerator = (kf * sympy.Mul(*(relcon[i.id] ** (s * h)
                                              for i, s in reactants)) -
                             kr * sympy.Mul(*(relcon[i.id] ** (s * h)
                                              for i, s in products)))
            elif version == 'hal':
                numerator = (kV * (sympy.sqrt(keq**h) *
                                   sympy.Mul(*(con[i.id] ** (s * h) for i, s in reactants)) -
                                   sympy.sqrt(keq**-h) *
                                   sympy.Mul(*(con[i.id] ** (s * h) for i, s in products))) /
                             sympy.Mul(*(kms[i.id] ** (sympy.Abs(s) * h)
                                         for i, s in species)))
            elif version == 'weg':
                numerator = (kV * (sympy.exp(-(h * sympy.Add(*(mus[i.id] * s
                                                               for i, s in species))) /
                                             (2 * R * T)) *
                                   sympy.Mul(*(con[i.id] ** (s * h) for i, s in reactants)) -
                                   sympy.exp(-(h * sympy.Add(*(mus[i.id] * s
                                                               for i, s in species))) /
                                             (2 * R * T)) *
                                   sympy.Mul(*(con[i.id] ** (s * h) for i, s in products))) /
                             sympy.Mul(*(kms[i.id] ** (sympy.Abs(s) * h)
                                         for i, s in species)))
            else:
                raise ValueError("Invalid numerator version for modular "
                                 "rate law: {}".format(version))

            # Construct denominator
            if rate_law == 'CM':
                denominator = (sympy.Mul(*((1 + relcon[i.id]) ** (s * h)
                                           for i, s in reactants)) +
                               sympy.Mul(*((1 + relcon[i.id]) ** (s * h)
                                           for i, s in products)) -
                               1)
            elif rate_law == 'SM':
                denominator = (sympy.Mul(*((1 + relcon[i.id]) ** (sympy.Abs(s) * h)
                                           for i, s in species)))
            elif rate_law == 'DM':
                denominator = (sympy.Mul(*(relcon[i.id] ** (s * h)
                                           for i, s in reactants)) +
                               sympy.Mul(*(relcon[i.id] ** (s * h)
                                           for i, s in products)) +
                               1)
            elif rate_law == 'FM':
                denominator = (sympy.Mul(*(relcon[i.id] ** (sympy.Abs(s) * h / 2)
                                           for i, s in species)))
            elif rate_law == 'PM':
                denominator = 1
            else:
                raise ValueError("Invalid denominator rate law for modular "
                                 "rate law:{}".format(rate_law))

            # Do the regulation first since there are extra parameters to add to the SBML.
            f_reg = 1
            d_reg = 0
            sbml_regulation_parameters = []
            for (s_id, constants, reg_type, reg_subtype) in regulation.get(reaction.id, []):
                # Create symbols (w, kia and optional b for partial regulation)
                c = con[s_id]
                if len(constants) == 2:
                    w = constants[0]
                    id_ = 'kia_{}_{}'.format(reaction.id, s_id)
                    kia = sympy.symbols(id_)
                    sbml_regulation_parameters.append((id_, constants[1]))
                if len(constants) == 3:
                    id_ = 'b_{}_{}'.format(reaction.id, s_id)
                    b = sympy.symbols(id_)
                    sbml_regulation_parameters.append((id_, constants[2]))
                if len(constants) not in (2, 3):
                    raise ValueError("Incorrect number of parameters for regulation.")

                # Add to SBML
                for pname, value in sbml_regulation_parameters:
                    p = kinetic_law.createParameter()
                    p.initDefaults()
                    p.setId(pname)
                    p.setName(pname)
                    p.setValue(value)

                if (reg_type, reg_subtype) == ('inhibition', 'simple'):
                    f_reg *= (kia / (kia + c)) ** w
                elif (reg_type, reg_subtype) == ('activation', 'simple'):
                    f_reg *= (c / (kia + c)) ** w
                elif (reg_type, reg_subtype) == ('inhibition', 'specific'):
                    d_reg += (c / kia) ** w
                elif (reg_type, reg_subtype) == ('activation', 'specific'):
                    d_reg += (kia / c) ** w
                elif (reg_type, reg_subtype) == ('inhibition', 'partial'):
                    f_reg *= (b + (1 - b) / (1 + c / kia)) ** w
                elif (reg_type, reg_subtype) == ('activation', 'partial'):
                    f_reg *= (b + (1 - b) * (c / kia) / (1 + c / kia)) ** w
                elif (reg_type, reg_subtype) == ('inhibition', 'complete'):
                    f_reg *= (1 / (1 + c / kia)) ** w
                elif (reg_type, reg_subtype) == ('activation', 'complete'):
                    f_reg *= ((c / kia) / (1 + c / kia)) ** w
                else:
                    raise ValueError("Unknown regulation type: {} - {}".format(reg_type,
                                                                               reg_subtype))

            # Total reaction
            final_expression = u * f_reg * numerator / (denominator + d_reg)

            # Update SBML
            ast = ASTNodeWrapper.from_sympy(final_expression).node
            exit = kinetic_law.setMath(ast)
            if exit == libsbml.LIBSBML_INVALID_OBJECT:
                raise ValueError('Constructed invalid rate law: {}.'.format(ast))

            reactions[reaction.id] = final_expression
        return reactions

    def getSIUnit(self, unit_id):
        """Get the unit expressed as a SI unit."""
        # Retrieve unit definition or error out.
        for i in range(self.sbml_object.getNumUnitDefinitions()):
            ud = self.sbml_object.getUnitDefinition(i)
            if unit_id == ud.id:
                break
        else:
            if unit_id:
                raise ValueError("{} not found!".format(unit_id))
            else:
                return ''

        kinds = {libsbml.UNIT_KIND_AMPERE: 'ampere',
                 libsbml.UNIT_KIND_BECQUEREL: 'becquerel',
                 libsbml.UNIT_KIND_CANDELA: 'candela',
                 libsbml.UNIT_KIND_CELSIUS: 'Celsius',
                 libsbml.UNIT_KIND_COULOMB: 'coulomb',
                 libsbml.UNIT_KIND_DIMENSIONLESS: '-',
                 libsbml.UNIT_KIND_FARAD: 'farad',
                 libsbml.UNIT_KIND_GRAM: 'gram',
                 libsbml.UNIT_KIND_GRAY: 'gray',
                 libsbml.UNIT_KIND_HENRY: 'henry',
                 libsbml.UNIT_KIND_HERTZ: 'hertz',
                 libsbml.UNIT_KIND_ITEM: 'item',
                 libsbml.UNIT_KIND_JOULE: 'joule',
                 libsbml.UNIT_KIND_KATAL: 'katal',
                 libsbml.UNIT_KIND_KELVIN: 'kelvin',
                 libsbml.UNIT_KIND_KILOGRAM: 'kilogram',
                 libsbml.UNIT_KIND_LITER: 'litre',
                 libsbml.UNIT_KIND_LITRE: 'litre',
                 libsbml.UNIT_KIND_LUMEN: 'lumen',
                 libsbml.UNIT_KIND_LUX: 'lux',
                 libsbml.UNIT_KIND_METER: 'metre',
                 libsbml.UNIT_KIND_METRE: 'metre',
                 libsbml.UNIT_KIND_MOLE: 'mole',
                 libsbml.UNIT_KIND_NEWTON: 'newton',
                 libsbml.UNIT_KIND_OHM: 'ohm',
                 libsbml.UNIT_KIND_PASCAL: 'pascal',
                 libsbml.UNIT_KIND_RADIAN: 'radian',
                 libsbml.UNIT_KIND_SECOND: 'second',
                 libsbml.UNIT_KIND_SIEMENS: 'siemens',
                 libsbml.UNIT_KIND_SIEVERT: 'sievert',
                 libsbml.UNIT_KIND_STERADIAN: 'steradian',
                 libsbml.UNIT_KIND_TESLA: 'tesla',
                 libsbml.UNIT_KIND_VOLT: 'volt',
                 libsbml.UNIT_KIND_WATT: 'watt',
                 libsbml.UNIT_KIND_WEBER: 'weber'}
        error = libsbml.UNIT_KIND_INVALID

        # Construct proper unit name from terms.
        name = []
        for i in range(ud.getNumUnits()):
            unit = ud.getUnit(i)
            try:
                SI_name = kinds[unit.kind]
            except ValueError:
                raise ValueError("Invalid unit kind: {}".format(unit.kind))
            scale = (10 ** unit.scale) * unit.multiplier
            exponent = unit.exponent

            if exponent != 1:
                SI_name = SI_name + '**{}'.format(exponent)
            if scale != 1:
                SI_name = '{}*'.format(scale) + SI_name
            name.append(SI_name)

        return ' * '.join(name)

    def update_annotation(self):
        """Update the annotation of all the reactions and species."""
        raise NotImplementedError()

    def writeSBML(self, path):
        """Write the SBML model document to the file in path."""
        exit = libsbml.writeSBMLToFile(self.document, path)
        if not exit:
            raise IOError('Could not write SBML file to {}'.format(path))


class SBMLObject(object):
    """Base wrapper for the libsbml species/reaction object."""

    sbml_object_type = 'Base'

    def __init__(self, sbml_object):
        """Create an libsbml wrapper object from a libsbml object.

        Arguments:
        sbml_object -- An SBML object as returned by libsbml.
        """
        self.sbml_object = sbml_object
        self.name = self.sbml_object.getName()
        self.id = self.sbml_object.getId()

        # Try to find all identifiers
        self.identifiers = []

        # Check URI section. TODO: Check which kind of relation it actually has.
        notes = []
        for i in range(self.sbml_object.getNumCVTerms()):
            cvterm = self.sbml_object.getCVTerm(i)
            for j in range(cvterm.getNumResources()):
                notes.append(cvterm.getResourceURI(j))

        # Check notes and annotation section
        if self.sbml_object.isSetNotes():
            notes.append(self.sbml_object.getNotesString())
        if self.sbml_object.isSetAnnotation():
            notes.append(self.sbml_object.getAnnotationString())
        self.parse_notes('\n'.join(notes))

        # Check for optional SBO terms
        sbo_term = self.sbml_object.getSBOTerm()
        if sbo_term != -1:
            self.identifiers.append((Identifiers.sbo, sbo_term))

    def update_annotation():
        """Update the annotation of the SBML object to save them when writing out the SBML."""
        raise NotImplementedError()

    def add_annotation():
        """Add new identifiers to the SBML object."""
        raise NotImplementedError()

    def parse_notes(self, notes):
        """Find stuff in notes that should have been an URI.

        Method for subclasses to overwrite. Will be called in __init__.
        """
        identifiers = []
        d = {"chebi": Identifiers.chebi_id,
             "kegg.compound": Identifiers.kegg_compound,
             "kegg.reaction": Identifiers.kegg_reaction,
             "pubchem.substance": Identifiers.pubchem_id,
             "ec-code": Identifiers.enzyme_EC,
             "reactome": Identifiers.reactome_reaction_id,
             "go": Identifiers.go,
             "rhea": Identifiers.rhea_id,
             "sabiork.reaction": Identifiers.sabiork_reaction_id,
             }
        # Start with urn and identifiers.org
        r = r"""urn:miriam:(.*?):(.*?)['"\s<]"""
        matches = re.findall(r, notes, re.IGNORECASE)
        r = r"""identifiers.org/(.*?)/(.*?)['"\s<]"""
        matches = matches + re.findall(r, notes, re.IGNORECASE)
        for name, value in matches:
            name = name.lower()
            if name.startswith('obo.'):
                name = name[4:]
            if name in d:
                name = d[name]
            # Best guess, could be improved to match our internal names better.
            # Anyway, never saw anything useful but the values in d yet.
            else:
                name = name + '_id'
            # Sometimes values get butchered by url encoding.
            if '%3A' in value:
                value.replace('%3A', ':')

            if value.startswith('CHEBI:'):
                value = value[6:]
            elif value.startswith('REACT_'):
                value = value[6:]
            identifiers.append((name, value))

        # Some extra regex based rules (Based on SBML files seen)
        # EC numbers in notes
        r = r"""EC Number: (.*?)['"\s<]"""
        matches = re.findall(r, notes, re.IGNORECASE)
        for value in matches:
            identifiers.append((Identifiers.enzyme_EC, value))
        # InChi in notes
        r = r"""(InChI=[0-9A-Za-z/,\-\+\(\)]+)</in:inchi>"""
        matches = re.findall(r, notes, re.IGNORECASE)
        for value in matches:
            identifiers.append((Identifiers.inchi, value))

        self.identifiers = list(set(identifiers))

    def __repr__(self):
        """String representation."""
        return "{}(id: {} - name: {})".format(self.sbml_object_type, self.id, self.name)

    def __str__(self):
        """Extended string representation."""
        s = "SBML {}: {} {{\n\tID: {}\n\tidentifiers:\n\t\t{}\n\t}}"
        return s.format(self.sbml_object_type, self.name, self.id,
                        '\n\t\t'.join([' - '.join(i) for i in self.identifiers]))


class Compartment(SBMLObject):
    """Wrapper for the libsbml compartment object."""

    def __init__(self, sbml_object):
        """Create the compartment wrapper object.

        Arguments:
        sbml_object -- An SBML object as returned by libsbml.
        """
        super(Compartment, self).__init__(sbml_object)
        self.sbml_object_type = 'Compartment'
        if self.sbml_object.isSetSize():
            self.size = self.sbml_object.getSize()
        else:
            self.size = None


class Species(SBMLObject):
    """Wrapper for the libsbml species object."""

    def __init__(self, sbml_object):
        """Create the species wrapper object.

        Arguments:
        sbml_object -- An SBML object as returned by libsbml.
        """
        super(Species, self).__init__(sbml_object)
        self.sbml_object_type = 'Species'
        self.compartment = self.sbml_object.getCompartment()

        self.initial_concentration = self.sbml_object.getInitialConcentration()


class Reaction(SBMLObject):
    """Wrapper for the libsbml reaction object."""

    def __init__(self, sbml_object):
        """Create the reaction wrapper object.

        Arguments:
        sbml_object -- An SBML object as returned by libsbml.
        """
        super(Reaction, self).__init__(sbml_object)
        self.sbml_object_type = 'Reaction'
        self.compartment = self.sbml_object.getCompartment()

        # Note that reactants and products have a stoichiometry, while
        # modifiers don't.
        reactants = [self.sbml_object.getReactant(i)
                     for i in range(self.sbml_object.getNumReactants())]
        products = [self.sbml_object.getProduct(i)
                    for i in range(self.sbml_object.getNumProducts())]
        modifiers = [self.sbml_object.getModifier(i)
                     for i in range(self.sbml_object.getNumModifiers())]

        # These will get updated by the main model from strings to actual species objects.
        self.species = {i.getSpecies() for i in reactants + products + modifiers}
        self.reactants = [(i.getSpecies(), i.getStoichiometry()) for i in reactants]
        self.products = [(i.getSpecies(), i.getStoichiometry()) for i in products]

        # For modifiers, SBO terms might be intersting
        # since they could give the type of interaction.
        self.modifier_SBO_terms = {}
        for i in modifiers:
            if i.isSetSBOTerm():
                self.modifier_SBO_terms[i.getSpecies()] = i.getSBOTerm()
        self.modifiers = [i.getSpecies() for i in modifiers]

        self.parameters = self.parse_kinetic_law()

    def parse_kinetic_law(self):
        """Parse the kinetic law for any parameter values - useful for Sabio-Rk SBML data."""
        law = self.sbml_object.getKineticLaw()
        ps = {}

        # Exit early if no kinetic law.
        if law == -1:
            return ps

        try:
            local_parameters = [law.getLocalParameter(i)
                                for i in range(law.getNumLocalParameters())]
        except AttributeError:
            local_parameters = []  # Early versions of SBML might not have parameters.

        try:
            parameters = [law.getParameter(i)
                          for i in range(law.getNumParameters())]
        except AttributeError:  # Early versions of SBML might not have parameters.
            parameters = []

        for p in parameters + local_parameters:
            x = {}
            if p.isSetValue():
                x['value'] = p.getValue()
                x['id'] = p.getId()
                x['name'] = p.getName()
                if p.getSBOTerm() != -1:
                    x['sbo'] = p.getSBOTerm()
                x['unit'] = p.getUnits()
                ps[x['id']] = x
        return ps


class ASTNodeWrapper(object):
    """Class for wrapping ASTNode of libsbml and converting it to Sympy and back."""

    functions = {
            libsbml.AST_FUNCTION,
            libsbml.AST_LAMBDA,
            }

    variables = {
            libsbml.AST_NAME,
            }

    numerical = {
            libsbml.AST_INTEGER,
            libsbml.AST_RATIONAL,
            libsbml.AST_REAL,
            libsbml.AST_REAL_E,
            }

    constants = {
            libsbml.AST_CONSTANT_E: sympy.E,
            libsbml.AST_CONSTANT_FALSE: sympy.false,
            libsbml.AST_CONSTANT_PI: sympy.pi,
            libsbml.AST_CONSTANT_TRUE: sympy.true,
            libsbml.AST_NAME_AVOGADRO: 6.02214086e23,
            }

    operators = {
            libsbml.AST_PLUS: operator.add,
            libsbml.AST_MINUS: operator.sub,
            libsbml.AST_POWER: operator.pow,
            libsbml.AST_FUNCTION_POWER: operator.pow,
            libsbml.AST_DIVIDE: operator.truediv,
            libsbml.AST_TIMES: operator.mul,
            }

    unary_operators = {
            libsbml.AST_PLUS: operator.pos,
            libsbml.AST_MINUS: operator.neg,
            }

    mathematical_functions = {
            libsbml.AST_FUNCTION_ABS: sympy.Abs,
            libsbml.AST_FUNCTION_ARCCOS: sympy.acos,
            libsbml.AST_FUNCTION_ARCCOSH: sympy.acosh,
            libsbml.AST_FUNCTION_ARCCOT: sympy.acot,
            libsbml.AST_FUNCTION_ARCCOTH: sympy.acoth,
            libsbml.AST_FUNCTION_ARCCSC: sympy.acsc,
            libsbml.AST_FUNCTION_ARCSEC: sympy.asec,
            libsbml.AST_FUNCTION_ARCSIN: sympy.asin,
            libsbml.AST_FUNCTION_ARCSINH: sympy.asinh,
            libsbml.AST_FUNCTION_ARCTAN: sympy.atan,
            libsbml.AST_FUNCTION_ARCTANH: sympy.atanh,
            libsbml.AST_FUNCTION_CEILING: sympy.ceiling,
            libsbml.AST_FUNCTION_COS: sympy.cos,
            libsbml.AST_FUNCTION_COSH: sympy.cosh,
            libsbml.AST_FUNCTION_COT: sympy.cot,
            libsbml.AST_FUNCTION_COTH: sympy.coth,
            libsbml.AST_FUNCTION_CSC: sympy.csc,
            libsbml.AST_FUNCTION_CSCH: sympy.csch,
            libsbml.AST_FUNCTION_EXP: sympy.exp,
            libsbml.AST_FUNCTION_FACTORIAL: sympy.factorial,
            libsbml.AST_FUNCTION_FLOOR: sympy.floor,
            libsbml.AST_FUNCTION_LN: sympy.ln,
            libsbml.AST_FUNCTION_PIECEWISE: sympy.Piecewise,
            libsbml.AST_FUNCTION_ROOT: sympy.root,
            libsbml.AST_FUNCTION_SEC: sympy.sec,
            libsbml.AST_FUNCTION_SECH: sympy.sech,
            libsbml.AST_FUNCTION_SIN: sympy.sin,
            libsbml.AST_FUNCTION_SINH: sympy.sinh,
            libsbml.AST_FUNCTION_TAN: sympy.tan,
            libsbml.AST_FUNCTION_TANH: sympy.tanh,
            }

    logical_functions = {
            libsbml.AST_LOGICAL_AND: sympy.And,
            libsbml.AST_LOGICAL_NOT: sympy.Not,
            libsbml.AST_LOGICAL_OR: sympy.Or,
            libsbml.AST_LOGICAL_XOR: sympy.Xor,
            }

    relational_functions = {
            libsbml.AST_RELATIONAL_EQ: sympy.Eq,
            libsbml.AST_RELATIONAL_GEQ: sympy.Ge,
            libsbml.AST_RELATIONAL_GT: sympy.Gt,
            libsbml.AST_RELATIONAL_LEQ: sympy.Le,
            libsbml.AST_RELATIONAL_LT: sympy.Lt,
            libsbml.AST_RELATIONAL_NEQ: sympy.Ne,
            }

    all_functions = {k: v for d in (operators,
                                    mathematical_functions,
                                    logical_functions,
                                    relational_functions)
                     for k, v in d.items()}

    not_implemented = {
            libsbml.AST_FUNCTION_ARCCSCH: "AST_FUNCTION_ARCCSCH",
            libsbml.AST_FUNCTION_ARCSECH: "AST_FUNCTION_ARCSECH",
            libsbml.AST_FUNCTION_DELAY: "AST_FUNCTION_DELAY",
            libsbml.AST_NAME_TIME: "AST_NAME_TIME",
            libsbml.AST_UNKNOWN: "AST_UNKNOWN",
            # defaults to 10, otherwise log base has to be given. How is this noted in the children?
            libsbml.AST_FUNCTION_LOG: "AST_FUNCTION_LOG",
            }

    def __init__(self, ASTNode):
        """Create an ASTNode wrapper from an ASTNode.

        Can also be created from a Sympy expression using ASTNodeWrapper.from_sympy."""
        self.node = ASTNode

    @classmethod
    def from_sympy(cls, sympy_expression):
        """Convert a sympy expression back to ASTNode through MathML."""
        # Create MathML
        mathml = sympy.printing.mathml(sympy_expression)
        # Add open and close tags and convert to ASCII for libsbml.
        math_open = "<math xmlns='http://www.w3.org/1998/Math/MathML'>"
        math_close = "</math>"

        mathml = (''.join((math_open, mathml, math_close))
                    .encode('ascii', 'xmlcharrefreplace'))

        # Check validity.
        ASTNode = libsbml.readMathMLFromString(mathml)
        if ASTNode is None or not ASTNode.isWellFormedASTNode:
            print(mathml)
            raise ValueError('Invalid MathML input to readMathMLFromString')

        return cls(ASTNode)

    def __iter__(self):
        """Iterate through the direct children nodes."""
        for i in range(self.node.getNumChildren()):
            yield ASTNodeWrapper(self.node.getChild(i))

    @property
    def type(self):
        """LibSBML node type."""
        return self.node.getType()

    @property
    def value(self):
        """Numerical value."""
        if self.type == libsbml.AST_INTEGER:
            return self.node.getInteger()
        elif self.type in self.numerical:
            return self.node.getReal()
        else:
            raise ValueError("Not a numerical value.")

    @property
    def name(self):
        """Variable of function name."""
        try:
            return self.node.getName()
        except AttributeError:
            raise ValueError("Not a named value")

    def to_sympy(self):
        """Recursively convert the AST to a Sympy representation."""
        if self.type in self.not_implemented:
            raise NotImplementedError('{} not implemented.'
                                      .format(self.not_implemented[self.type]))
        elif self.type in self.constants:
            return self.constants[self.type]
        elif self.type in self.variables:
            return sympy.Symbol(self.name)
        elif self.type in self.numerical:
            return self.value
        elif self.type in self.all_functions or self.type in self.functions:
            children = [child.to_sympy() for child in self]
            if len(children) == 1 and self.type in self.operators:
                    return self.unary_operators[self.type](*children)
            elif self.type in self.all_functions:
                return self.all_functions[self.type](*children)
            else:
                if self.type == libsbml.AST_LAMBDA:
                    return sympy.Lambda(children[:-1], children[-1])
                elif self.type == libsbml.AST_FUNCTION:
                    return sympy.Function(self.name)(*children)
                else:
                    raise ValueError("Unsupported ASTNode type: {}".format(self.type))

    def __str__(self):
        return "ASTNodeWrapper({})".format(libsbml.formulaToString(self.node))

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    def naive_jacobian(f, u, eps=1e-6):
        """Evaluate partial derivatives of f(u) numerically.

        :note:
            This routine is currently naive and could be improved.
        :returns:
            (*f.shape, *u.shape) array ``df``, where df[i,j] ~= (d f_i / u_j)(u)

        Sourced from the scipy mailing list from Pauli Virtanen.
        """
        f0 = np.asarray(f(u))
        u_shape = u.shape
        nu = np.prod(u_shape)
        f_shape = f0.shape
        nf = np.prod(f_shape)
        df = np.empty([nf, nu])
        for k in range(nu):
            du = np.zeros(nu)
            du[k] = max(eps*abs(u.flat[k]), eps)
            f1 = np.asarray(f(u + np.reshape(du, u_shape)))
            df[:, k] = np.reshape((f1 - f0) / eps, [nf])
        df.shape = f_shape + u_shape
        return df

    def approx_jacobian(x, func, epsilon, *args):
        """Approximate the Jacobian matrix of callable function func

           * Parameters
             x       - The state vector at which the Jacobian matrix is desired
             func    - A vector-valued function of the form f(x,*args)
             epsilon - The perturbation used to determine the partial derivatives
             *args   - Additional arguments passed to func

           * Returns
             An array of dimensions (lenf, lenx) where lenf is the length
             of the outputs of func, and lenx is the number of

           * Notes
             The approximation is done using forward differences

        Sourced from the scipy mailing list from Rob Falck.
        """
        x0 = np.asarray(x)
        f0 = func(*((x0,)+args))
        jac = np.zeros([len(x0), len(f0)])
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0
        return jac.transpose()

    logging.basicConfig(level=logging.INFO)
    import time
    # working_dir = os.getcwd()
    # model_dir = os.path.join(working_dir, 'test_data', 'models')
    # test_model_paths = []
    # test_model_paths.append(os.path.join(model_dir, 'Costa2014.xml'))
    # test_model_paths.append(os.path.join(model_dir, 'Chassagnole2012.xml'))
    # test_model_paths.append(os.path.join(model_dir, 'Hynne2001.xml'))

    # models = []
    # for model_path in test_model_paths:
    #     models.append(Model(model_path))

    # model_path = 'test_data/models/Millard2017.xml'
    # model_path = 'test_data/models/Chassagnole2012.xml'
    # model_path = 'test_data/models/Hynne2001.xml'
    # model_path = 'test_data/models/Emiola2014_fixed_rules.xml'
    # model_path = 'test_data/models/Costa2014.xml'
    # model_path = 'test_data/models/Jahan2016_fixed.xml'
    # model_path = 'test_data/models/Jahan2016v22.xml'
    # model_path = 'test_data/models/Kesten2015_annotated.xml'
    model_path = 'test_data/example/testing/iJO1366/compressed/model_kinetics.xml'

    m = Model(model_path)
    m.run_initial_assignments()
    m.inline_functions()
    m.inline_rules()

    keys = list(m.species.keys())

    atol = 1e-12
    rtol = 1e-8

    import roadrunner as rr

    m_rr = rr.RoadRunner(model_path)
    m_rr.integrator.relative_tolerance = rtol
    m_rr.integrator.absolute_tolerance = atol
    m_rr.timeCourseSelections = ['[{}]'.format(k) for k in keys]

    t0 = time.time()
    try:
        rr_points = m_rr.simulate(start=0, end=10, steps=10000)
        print("Successfully simulated with Roadrunner (t={:.2f}).".format(time.time() - t0))
    except RuntimeError:
        rr_points = np.zeros(1)
        print("Failed to simulate with Roadrunner (t={:.2f}).".format(time.time() - t0))

    t0 = time.time()
    jac = False
    # Use the experimental sympy to LLVM implementation
    # Note: this generates fast code, but takes a while to generate
    #       (because of the CSE/simplify steps?).
    # Max/Heaviside is not implemented so we cannot enforce reversibility.

    jit = False
    enforce_reversibility = False
    compensate_volumes = False
    disabled_metabolite_fluxes = None
    if jac:
        f, jac = m.get_ode_function(disabled_metabolite_fluxes, enforce_reversibility,
                                    compensate_volumes, jac)
    else:
        f = m.get_ode_function(disabled_metabolite_fluxes, enforce_reversibility,
                               compensate_volumes, jac)
    y = m.get_state(array=True)
    print("Created functions for model (t={:.2f}).".format(time.time() - t0))

    import scipy.integrate

    integrator = 'lsoda'
    nsteps = 5e3
    if jac:
        r = scipy.integrate.ode(f, jac).set_integrator(integrator,
                                                       atol=atol,
                                                       rtol=rtol,
                                                       nsteps=nsteps,
                                                       )
    else:
        r = scipy.integrate.ode(f).set_integrator(integrator,
                                                  atol=atol,
                                                  rtol=rtol,
                                                  nsteps=nsteps,
                                                  )
    r.set_initial_value(y, 0.0)
    t = 10
    dt = 0.001

    points = []
    timepoints = []
    values = y

    t0 = time.time()
    while r.successful() and r.t < t:
        timepoints.append(r.t)
        points.append(values)
        values = r.integrate(r.t + dt)

    points = np.vstack(points)
    t = np.array(timepoints)
    if t.size > 1:
        print("Successfully simulated with Scipy (t={:.2f}).".format(time.time() - t0))
    else:
        print("Failed to simulate with Scipy (t={:.2f}).".format(time.time() - t0))

    import matplotlib.pyplot as plt
    plt.figure("Roadrunner (CVODE)")
    plt.plot(rr_points)

    plt.figure("Scipy ({})".format(integrator))
    plt.plot(t, points)

    plt.show()

    # TODO: Rewrite into separate classes for SBML model and integrable model with
    # clear inter-conversions.
