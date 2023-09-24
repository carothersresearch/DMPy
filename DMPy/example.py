#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Complete example script to reproduce all results in the paper.

Author: Rik van Rosmalen
"""
# Python 2/3 compatibility
from __future__ import print_function
from __future__ import division

# Standard library imports
import os
import sys
import shutil

import time
import random

import functools
import collections

import json
import inspect

# Standard scientific imports
import numpy as np

# Own library imports
import pipeline
import balancer
import sbmlwrap
import random_model
import simulate

# Since python 3.4 pathlib is a standard library module so check if it exists already..
if (sys.version_info > (3, 4)):
    import pathlib
else:
    # Note: The maintained backport is called pathlib2, not pathlib!
    import pathlib2 as pathlib

# Step 0:
#   - Setup working directory for (intermediate) results
#   - Configure the automatic skipping of unchanged steps and logging of settings.
# timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
working_directory_path = pathlib.Path('./test_data/example')
working_directory_path.mkdir(parents=True, exist_ok=True)

# We define this before the function so that the decorator works as expected.
check = True
silent = False
metadata_path = working_directory_path.joinpath('metadata.json')


def check_for_refresh(check, metadata_path, ignore=None, silent=True):
    """Check the metadata file for previous runs to see if the result is still valid.

    Assumptions that are made here:
       - All file path arguments are pathlib (Pure)Path objects.
       - The function returns a (series of) pathlib (Pure)Path object as output.
       - If an argument should be ignored, include it in ignore.

    The format of the metadata file (json) is the following:
       {function:
           {
           'settings':
               {arg: value for arg, value in args}},
           'input_paths':
               {path: timestamp(stamp) for path in args},
           'output_paths':
               [path,
                path,
                ...]
           },
        function:
            {....},
        ....}

    If the timestamps or the arguments don't match, the function is executed and the metadata is
    updated. If everything matches, the previous output path is returned without executing the
    decorated function.

    As a bonus, this keeps a log of the exact settings that made the current result files,
    given that the scripts that generated them are not modified in between.

    Note: Settings saved in a tuple will be loaded from json as a list, thus causing a mismatch.

    # Possible TODO #
    - Incorporate git status to watch for code changes? Or manually hash files? Dependencies will
      be a annoying.
    """
    if ignore is None:
        ignore = set()

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            # Ignore the whole thing
            if not check:
                return function(*args, **kwargs)
            # Pair any non-keyword args with their names so we gather everything in kwargs.
            # This makes downstream processing a bit easier.
            if args:
                argnames = inspect.getargspec(function).args
                kwargs.update(dict(zip(argnames, args)))

            refresh = False
            dirty = []
            # Try to open the file with the metadata (json formatted)
            try:
                with open(str(metadata_path), 'r') as metadata_file:
                        metadata = json.load(metadata_file)
            except (IOError, ValueError):
                # IOError - file doesn't exist
                # ValueError - Couldn't parse json
                # Whatever the reason, we need to start from scratch.
                metadata = {}
                refresh = True
                dirty.append('metadata.json not found or unreadable.')

            # Try to open the metadata specific to the function called.
            try:
                f_metadata = metadata[function.func_name]
            except KeyError:
                # The function was not called and saved before, so start fresh for this function.
                f_metadata = {}
                refresh = True
                dirty.append('Function not found in metadata.json')

            # Check arguments for changes
            in_paths = {}
            for arg, value in kwargs.items():
                # Check if we should skip this one.
                if arg in ignore:
                    continue
                # If it is a path we have to check the timestamps of the file to see if it changed.
                if isinstance(value, pathlib.PurePath):
                    value = str(value)
                    # Else, generate a new timestamp and compare to old.
                    # print(arg, value, ignore)
                    try:
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',
                                                  time.localtime(os.path.getmtime(value)))
                    except OSError:
                        # File doesn't exist.
                        refresh = True
                        dirty.append('File not found: {}'.format(value))
                        raise

                    if not refresh:
                        try:
                            if timestamp != f_metadata['input_paths'][value]:
                                # If the timestamp is different, mark for refresh.
                                refresh = True
                                dirty.append('Input time stamp mismatch for: {}'.format(value))
                        except KeyError:
                            # If the output file wasn't mentioned before, mark for refreshing.
                            refresh = True
                            dirty.append('Input not seen yet: {}'.format(value))
                    # Save all the paths separately since we want to save the timestamps,
                    # and not just the values.
                    in_paths[value] = timestamp
                # For non-paths, compare to the actual value
                else:
                    try:
                        if value != f_metadata['settings'][arg]:
                            # If the argument is different, mark for refreshing.
                            # Note that when we save a tuple to json, it will be returned
                            # as a list and not compare equal.
                            refresh = True
                            dirty.append('Setting mismatch for: {}'.format(arg))
                            # print(f_metadata['settings'][arg])
                            # print(value)
                    except KeyError:
                        # If the argument didn't exist yet, mark for refreshing.
                        refresh = True
                        dirty.append('Setting not seen yet: {}'.format(arg))

            # If everything is still good after all checks
            # return the previous path and quit.
            if not refresh:
                out = [pathlib.Path(i) for i in f_metadata['output_paths']]
                # Final check - outputs still exist?
                for path in out:
                    if not os.path.exists(str(path)):
                        refresh = True
                        dirty.append('Missing output file: {}'.format(path))

                if not refresh:
                    if not silent:
                        print("Skipped function: {}".format(function.func_name))
                        if len(out) == 1:
                            return out[0]
                        else:
                            return out

            # If needed we can print some info about the reason
            if not silent:
                print("Re-running function: {}".format(function.func_name))
                for i in dirty:
                    print('\t', i)
                    if i in ('metadata.json not found or unreadable.',
                             'Function not found in metadata.json'):
                        break

            # Else we require an update. (Take timings for debugging.)
            t0 = time.time()
            out = function(**kwargs)
            timing = time.time() - t0
            # Check to see if we have only one or more paths and put them in a
            # list so we can treat it the same later.
            if isinstance(out, pathlib.Path):
                paths_out = [out]
            else:
                paths_out = list(out)

            # Make sure to update the metadata for next time.
            last_updated = time.strftime('%Y-%m-%d %H:%M:%S')
            f_metadata = {'settings': {k: v for k, v in kwargs.items()
                                       if (str(v) not in in_paths
                                           and k not in ignore)},
                          'input_paths': in_paths,
                          'output_paths': [str(i) for i in paths_out],
                          'meta_last_updated': last_updated,
                          'meta_execution_time': timing}
            # Update the last modified date of the complete metadata.
            metadata['meta_last_updated'] = last_updated
            # Keep the old metadata and override the current function part only.
            metadata[function.func_name] = f_metadata
            # print("New:", metadata)
            # Override the complete file again, although only one function is
            # changed this is more simple then modifying the file in place.
            with open(str(metadata_path), 'w') as metadata_file:
                json.dump(metadata, metadata_file, indent=4)

            return out

        return wrapper

    return decorator


def load_parameters(parameter_path):
    """Load a parameter tab separated file.

    The format should be (per line):
        parameter type, reaction id, species id, mean, std
    Extra columns will be ignored.

    '', NaN', 'nan' or 'None' values will be replaced with None
    """
    missing = set(('', 'NaN', 'nan', 'None'))
    parameters = []
    with open(str(parameter_path), 'r') as parameter_file:
        parameter_file.readline()  # Skip header
        for line in parameter_file:
            line = line.strip().split('\t')[:5]
            t = line[0]
            r = None if line[1] in missing else line[1]
            s = None if line[2] in missing else line[2]
            m = None if line[3] in missing else float(line[3])
            sd = None if line[4] in missing else float(line[4])
            parameters.append((t, r, s, m, sd))
    return parameters


def save_parameters(parameter_path, parameters):
    """Save a set of parameters as a tsv file."""
    with open(str(parameter_path), 'w') as outfile:
        outfile.write('QuantityType\tSBMLReactionID\tSBMLSpeciesID\tMean\tStd\n')
        for t, r, m, v, sd in sorted(parameters):
            outfile.write('\t'.join(str(i) for i in (t, r, m, v, sd)))
            outfile.write('\n')


@check_for_refresh(check, metadata_path, ignore=['to_path'], silent=silent)
def copy_model(from_path, to_path):
    """Copy a model file to the working directory."""
    shutil.copy(str(from_path), str(to_path))
    return to_path


@check_for_refresh(check, metadata_path, ignore=['to_path'], silent=silent)
def generate_model(to_path, n_metabolites, n_reactions, n_compartments,
                   n_regulators, max_degree, gamma, regulatory_types,
                   C_volumes=None, random_seed=None):
    """Generate a random metabolic model and write it to an SBML file."""
    # Generate the stoichiometry and regulation matrix and the compartment array.
    S, C, R = random_model.generate_stoichiometry(n_metabolites, n_reactions, n_compartments,
                                                  n_regulators, max_degree, gamma,
                                                  regulatory_types, random_seed=None)

    # Convert to a model structure
    N = random_model.Network.from_stoichiometry(S, C=C, R=R, C_volumes=C_volumes)

    # Convert to an SBML string and save. [Note that this does not contain kinetics/parameters]
    with open(str(to_path), 'w') as model_file:
        model_file.write(N.to_sbml())
    return to_path


def generate_regulation(model_path, distributions, random_seed=None):
    """Generate regulation for a (random) model annotated by SBO terms for each modifier."""
    if random_seed:
        np.random.seed(random_seed)
    model = sbmlwrap.Model(str(model_path))
    modifiers = {}
    for name, reaction in model.reactions.items():
        for metabolite in reaction.modifiers:
            # reactants = set(i[0] for i in reaction.reactants + reaction.products)
            # if metabolite not in reactants:
            SBO = reaction.modifier_SBO_terms.get(metabolite.id, None)
            if name in modifiers:
                modifiers[name].append((metabolite.id, SBO))
            else:
                modifiers[name] = [(metabolite.id, SBO)]

    distributions = distributions.copy()
    for i in 'ki', 'ka':
        d_type, mu, sigma = distributions[i]
        if d_type.lower() == 'normal':
            distributions[i] = random_model.NormalDistribution(mu, sigma)
        elif d_type.lower() == 'lognormal':
            distributions[i] = random_model.LogNormalDistribution(mu, sigma)

    regulation = collections.defaultdict(list)
    for reaction, reaction_modifiers in modifiers.items():
        for modifier, sbo in reaction_modifiers:
            regulation_type, regulation_subtype = random_model.SBO_kinetics[SBO]
            if 'inhibition' in regulation_type:
                kia = distributions['ki'].draw()
            elif 'activation' in regulation_type:
                kia = distributions['ka'].draw()
            w = 1
            if 'partial' in regulation_subtype:
                parameters = [w, kia, .25 + np.random.random() * 0.75]
            else:
                parameters = [w, kia]
            regulation[reaction].append([modifier, parameters,
                                         regulation_type, regulation_subtype])
    return dict(regulation)


@check_for_refresh(check, metadata_path, silent=silent)
def generate_kinetics(model_path, parameter_path, ignore_reactions,
                      regulation, cooperativities, rate_law, version,
                      ignore_concentrations=False):
    """Generate kinetics and update the model, including parameters."""
    model = sbmlwrap.Model(str(model_path))
    model.kineticize(str(parameter_path), regulation, rate_law, version,
                     cooperativities, ignore_reactions,
                     ignore_concentrations=ignore_concentrations)
    kinetic_model_path = model_path.parent.joinpath('model_kinetics.xml')
    model.writeSBML(str(kinetic_model_path))
    return kinetic_model_path


@check_for_refresh(check, metadata_path, ignore=['to_path'], silent=silent)
def save_distributions(to_path, distributions):
    """Save distribution objects."""
    s = '\n'.join('{}: {}'.format(k, v) for k, v in distributions.items())
    with open(str(to_path), 'w') as distribution_file:
        distribution_file.write(s)
    return to_path


@check_for_refresh(check, metadata_path, ignore=['output_dir'], silent=silent)
def find_parameters(model_path, organism, output_dir, additional_parameter_path=None,
                    current_concentration_error=1e-6):
    """Find parameters using the pipeline."""
    pipeline.main(str(model_path), organism, str(output_dir), databases='all',
                  method='default', stopat='balance', skipbrendalogin=False,
                  task_dump=True)

    parameters_out = output_dir.joinpath('parameters.tsv')

    # Load parameter file.
    parameters = load_parameters(parameters_out)
    # Load extra if we have it.
    if additional_parameter_path is not None:
        additional_parameters = load_parameters(additional_parameter_path)
        parameters.extend(additional_parameters)

    # Retrieve original concentrations and add with extremely small error so they stay conserved
    # as much as possible
    for species in sbmlwrap.Model(str(model_path)).species.values():
        c = species.initial_concentration
        # Use small values instead of 0 since we'll move to log scale later.
        if c == 0:
            c = 1e-3
            print('Concentration of {} set from 0 to {} to avoid downstream errors'
                  .format(species.id, c))
        parameters.append(('concentration', None, species.id, c,
                           c * current_concentration_error))

    # Convert long type names to short.
    mapping = {
           'Michaelis constant': 'km',
           'inhibitory constant': 'ki',
           'activation constant': 'ka',
           'equilibrium constant': 'keq',
           'catalytic rate constant geometric mean': 'kv',
           'concentration of enzyme': 'u',
           'substrate catalytic rate constant': 'kcat',
           'product catalytic rate constant': 'kcat-',
           'forward maximal velocity': 'vmax',
           'reaction affinity': 'A',
           'standard chemical potential': 'mu*',
           'chemical potential': 'mu',
           'concentration': 'c',
           }

    parameters_mapped = []
    for (t, r, s, m, sd) in parameters:
        parameters_mapped.append((mapping[t], r, s, m, sd))

    # Move parameters out file
    parameters_out_pipeline = output_dir.joinpath('parameters_pipeline.tsv')
    shutil.copy(str(parameters_out), str(parameters_out_pipeline))

    save_parameters(parameters_out, parameters_mapped)

    return parameters_out, parameters_out_pipeline


def loadDistributions(distribution_path):
    """Load distributions into distribution objects."""
    distributions = {}
    with open(str(distribution_path), 'r') as distribution_file:
        for line in distribution_file:
            key, args = line.strip().split(': ')
            distribution_type, mu, sigma = args.strip('([])').split(', ')
            distribution_type = distribution_type.strip('\'\"').lower()
            if distribution_type == 'normal':
                distributions[key] = random_model.NormalDistribution(float(mu), float(sigma))
            elif distribution_type == 'lognormal':
                distributions[key] = random_model.LogNormalDistribution(float(mu), float(sigma))
            else:
                raise ValueError("Unknown distribution type: {}".format(distribution_type))
    return distributions


@check_for_refresh(check, metadata_path, ignore=['to_dir'], silent=silent)
def generate_parameters(model_path, distribution_path, to_dir, noise, random_seed=None):
    """Generate a random set of parameters for a model using the distributions."""
    model = random_model.Network.from_sbml_file(str(model_path))
    distributions = loadDistributions(distribution_path)

    model.update_distributions(**distributions)
    parameters = model.parameterize(random_seed=random_seed)
    noisy_parameters = model.generate_parameter_measurements(noise, random_seed=random_seed)

    true_parameter_path = to_dir.joinpath('true_parameters.tsv')
    noisy_parameter_path = to_dir.joinpath('data_parameters.tsv')

    save_parameters(true_parameter_path, parameters)
    save_parameters(noisy_parameter_path, noisy_parameters)
    return true_parameter_path, noisy_parameter_path


def filter_parameter_data(parameters, sd_filler):
    """Missing standard deviations will be filled with sd_filler * mean.

    sd_filler can either be an float that will be applied to all types,
    or a dictionary of parameter type: sd_filler."""
    new_parameters = []
    if not isinstance(sd_filler, collections.Mapping):
        filler = collections.defaultdict(lambda: sd_filler)
    else:
        filler = sd_filler

    for t, r, s, m, sd in parameters:
        if sd is None:
            sd = filler[t] * m
        new_parameters.append((t, r, s, m, sd))
    return new_parameters


def model_properties(model):
    """Retrieve some convenient model properties."""
    S = model.get_stoichiometry_matrix()

    r_pos = collections.OrderedDict(((j, i) for i, j in enumerate(model.reactions.keys())))
    reactions = list(r_pos.keys())

    s_pos = collections.OrderedDict(((j, i)for i, j in enumerate(model.species.keys())))
    compounds = list(s_pos.keys())
    return S, r_pos, reactions, s_pos, compounds


@check_for_refresh(check, metadata_path, ignore=['to_path'], silent=silent)
def balance_parameters(model_path, parameter_path, distribution_path, to_path, ignore_reactions,
                       augment=True, sd_filler=0.1, parameter_limits=None):
    """Use the algorithm of Lubitz et al. to balance a set of parameters."""
    if parameter_limits is None:
        parameter_limits = {}

    distributions = loadDistributions(distribution_path)

    # Parameters
    parameters = filter_parameter_data(load_parameters(parameter_path), sd_filler)

    # Model
    model = sbmlwrap.Model(str(model_path))
    S, r_pos, reactions, s_pos, compounds = model_properties(model)

    ignore_species = []
    if ignore_reactions:
        # Remove ignored reactions so we don't include them in the balancing procedure.
        reactions = [i for i in reactions if i not in ignore_reactions]
        ignore_r_indeces = []
        dirty_species = set()
        for i in ignore_reactions:
            dirty_species |= model.reactions[i].species
            ignore_r_indeces.append(r_pos[i])

        ignore_s_indeces = []
        for i in dirty_species:
            for r in reactions:
                # Match? -> Break and keep
                if i in model.reactions[r].species:
                    break
            # No break? -> Remove
            else:
                ignore_s_indeces.append(s_pos[i.id])
                ignore_species.append(i.id)

        # Create a mask for the S matrix
        mask = np.ones(S.shape, dtype=bool)
        mask[:, ignore_r_indeces] = False
        mask[ignore_s_indeces, :] = False

        # Refresh compounds and reactions positions
        OD = collections.OrderedDict
        r_pos = OD((j, i) for i, j in enumerate(k for k in model.reactions.keys()
                                                if k not in ignore_reactions))
        s_pos = OD((j, i) for i, j in enumerate(k for k in model.species.keys()
                                                if k not in ignore_species))

        # Refresh compounds and reactions
        compounds = list(s_pos.keys())
        reactions = list(r_pos.keys())

        # Filter and reshape S
        S = S[mask].reshape(len(s_pos), len(r_pos))

    # Priors expects parameter type: (mean, sd)
    priors = {k: v.priors for k, v in distributions.items()}

    # Data is expected in a different order.
    data = []
    for (pt, r, m, mean, sd) in parameters:
        # Can't use data that we just ignored.
        if (r not in r_pos and r is not None) or (m not in s_pos and m is not None):
            print("Ignored value (reaction or substrate ignored):", pt, r, m, mean, sd)
            continue

        # Clip to limits if defined.
        if pt in parameter_limits:
            low, high = parameter_limits[pt]
            old = (pt, r, m, mean, sd)
            if mean > high:
                print("Limited value:", pt, r, m, mean, 'to', high, .1 * high)
                mean = high
                sd = .1 * high
            elif mean < low:
                print("Limited value:", pt, r, m, mean, 'to', low, .1 * low)
                mean = low
                sd = .1 * low

        # SD of 0 can cause a singular matrix error when inverting the covariance matrix.
        if sd != 0:
            data.append((mean, sd, pt, m, r))
        else:
            data.append((mean, 1e-12, pt, m, r))

    result = balancer.balance(compounds, reactions, S, data, priors,
                              balancer.dependencies, balancer.nonlog,
                              balancer.R, balancer.T, augment,
                              s_pos, r_pos)
    npz_path = to_path.with_suffix('.npz')
    result.save(str(npz_path))

    tsv_path = to_path.with_suffix('.tsv')
    mapping = {
               'km': 'Michaelis constant',
               'ki': 'inhibitory constant',
               'ka': 'activation constant',
               'keq': 'equilibrium constant',
               'kv': 'catalytic rate constant geometric mean',
               'u': 'concentration of enzyme',
               'kcat': 'substrate catalytic rate constant',
               'kcat-': 'product catalytic rate constant',
               'vmax': 'forward maximal velocity',
               'A': 'reaction affinity',
               'mu*': 'standard chemical potential',
               'mu': 'chemical potential',
               'c': 'concentration',
               }

    with open(str(tsv_path), 'w') as outfile:
        outfile.write('QuantityType\tSBMLReactionID\tSBMLSpeciesID\tMean\tStd\n')
        for (t, m, r), mean, sd in zip(result.columns, result.median, result.sd):
            t = mapping[t]
            x = [i if i is not None else '' for i in (t, r, m, mean, sd)]
            outfile.write('\t'.join(str(i) for i in x))
            outfile.write('\n')

    return npz_path, tsv_path


def resample_error(balancer_result, error=1, columns=None, conserved=None):
    """Re-sample parameter distribution using the sigma as a new dataset."""
    if conserved is None:
        override = []
    else:
        if columns is None:
            raise ValueError('Requires columns to determine conserved species!')
        override = [i for i, (p, c, r) in enumerate(columns) if p in conserved]

    sample = balancer_result.sample(np.random.normal(0, error, balancer_result.q_post.shape))
    # Override values that should not be changed.
    sample[override] = balancer_result.median[override]
    return sample


def resample_fraction(mu, sigma, columns, fraction, keep_parameters=None):
    """Re-sample a fraction of the parameters as a new dataset."""
    if keep_parameters is None:
        keep_parameters = []
    # Here we save conserved parameters separately if wanted.
    temp_conserved = []
    # Here we save everything else.
    temp = []
    for m, s, (p, c, r) in zip(mu, sigma, columns):
        if p in keep_parameters:
            temp_conserved.append((m, s, p, c, r))
        else:
            temp.append((m, s, p, c, r))
    # Return conserved and a sample of the rest based on fraction.
    return temp_conserved + random.sample(temp, int(fraction * len(temp)))


class ParameterGenerator(object):
    """Class used for generating parameter sets with specified settings from balancing results."""

    def __init__(self, balanced_result_path, distribution_path, model_path,
                 sampling_distribution_with, exclude_from_resampling_of_fraction,
                 exclude_from_resampling_with_error, augment=True):
        model = sbmlwrap.Model(str(model_path))
        self.S, self.r_pos, self.reactions, self.s_pos, self.compounds = model_properties(model)
        self.priors = {k: v.priors for k, v in loadDistributions(distribution_path).items()}

        self.balanced_reference = balancer.BalancingResult.load(str(balanced_result_path))

        self.sampling_distribution_with = sampling_distribution_with
        self.exclude_from_resampling_of_fraction = exclude_from_resampling_of_fraction
        self.exclude_from_resampling_with_error = exclude_from_resampling_with_error
        self.augment = augment

    def samples(self, fractions, fraction_resamples, distribution_resamples):
        """Return an array of parameter sets.

        The array will have shape: (n_fractions, n_fraction_resamples,
                                    n_distributions_resamples, n_parameters)
        """
        size = (len(fractions), fraction_resamples,
                distribution_resamples, len(self.balanced_reference.columns))
        sets = np.zeros(size, dtype=np.float64)
        for fi, fraction in enumerate(fractions):
            for fs in range(fraction_resamples):
                # Sample a fraction of complete parameter distribution set
                new_data = resample_fraction(self.balanced_reference.median,
                                             self.balanced_reference.sd,
                                             self.balanced_reference.columns, fraction,
                                             self.exclude_from_resampling_of_fraction)

                # Balance set again to obtain a new complete distribution set
                rebalanced_fraction = balancer.balance(self.compounds, self.reactions, self.S,
                                                       new_data, self.priors,
                                                       balancer.dependencies, balancer.nonlog,
                                                       balancer.R, balancer.T, self.augment,
                                                       self.s_pos, self.r_pos)

                for ds in range(distribution_resamples):
                    # Sample individual parameter sets from the distributions
                    final_parameters = resample_error(rebalanced_fraction,
                                                      self.sampling_distribution_with,
                                                      self.balanced_reference.columns,
                                                      self.exclude_from_resampling_with_error)
                    # Save sets into array that will be saved to a file later.
                    sets[fi, fs, ds] = final_parameters
        return sets

    def sample_generator(self, fraction):
        """Lazy generator to yield  samples at the fraction and settings specified.

        Yields from a single fractional sample of the reference parameters. To
        resample the fraction, reinitialize the generator."""
        # Sample a fraction of complete parameter distribution set
        new_data = resample_fraction(self.balanced_reference.median,
                                     self.balanced_reference.sd,
                                     self.balanced_reference.columns, fraction,
                                     self.exclude_from_resampling_of_fraction)

        # Balance set again to obtain a new complete distribution set
        rebalanced_fraction = balancer.balance(self.compounds, self.reactions, self.S,
                                               new_data, self.priors,
                                               balancer.dependencies, balancer.nonlog,
                                               balancer.R, balancer.T, self.augment,
                                               self.s_pos, self.r_pos)

        while True:
            # Sample individual parameter sets from the distributions
            final_parameters = resample_error(rebalanced_fraction,
                                              self.sampling_distribution_with,
                                              self.balanced_reference.columns,
                                              self.exclude_from_resampling_with_error)
            # Yield single iteration. To restart, call the function anew.
            yield final_parameters


class ParameterGenerator_Direct_Input(object):
    """Class used for generating parameter sets with specified settings from
    direct parameter input."""

    def __init__(self, data_input_path, distribution_path, model_path, balanced_result_path,
                 sampling_distribution_with, exclude_from_resampling_of_fraction,
                 exclude_from_resampling_with_error, augment=True, stderror=0.1):
        model = sbmlwrap.Model(str(model_path))
        self.S, self.r_pos, self.reactions, self.s_pos, self.compounds = model_properties(model)
        self.priors = {k: v.priors for k, v in loadDistributions(distribution_path).items()}
        self.sampling_distribution_with = sampling_distribution_with
        self.exclude_from_resampling_of_fraction = exclude_from_resampling_of_fraction
        self.exclude_from_resampling_with_error = exclude_from_resampling_with_error
        self.augment = augment

        self.balanced_reference = balancer.BalancingResult.load(str(balanced_result_path))

        # Read in data
        parameters = load_parameters(data_input_path)
        self.mu, self.sigma, self.columns = [], [], []
        for (t, r, s, m, sd) in parameters:
            self.mu.append(m)
            if sd is None:
                sd = m * 0.1
            self.sigma.append(sd)
            # Note the switch of the reaction and substrate order
            self.columns.append((t, s, r))

    def sample_generator(self, fraction):
        new_data = resample_fraction(self.mu, self.sigma, self.columns, fraction,
                                     self.exclude_from_resampling_of_fraction)
        # Balance set again to obtain a new complete distribution set
        rebalanced_fraction = balancer.balance(self.compounds, self.reactions, self.S,
                                               new_data, self.priors,
                                               balancer.dependencies, balancer.nonlog,
                                               balancer.R, balancer.T, self.augment,
                                               self.s_pos, self.r_pos)

        while True:
            # Sample individual parameter sets from the distributions
            final_parameters = resample_error(rebalanced_fraction,
                                              self.sampling_distribution_with,
                                              rebalanced_fraction.columns,
                                              self.exclude_from_resampling_with_error)
            # Yield single iteration. To restart, call the function anew.
            yield final_parameters


@check_for_refresh(check, metadata_path, ignore=['to_path'], silent=silent)
def generate_parameter_sets(balanced_result_path, distribution_path, model_path, to_path,
                            fractions, fraction_resamples, distribution_resamples,
                            sampling_distribution_with,
                            exclude_from_resampling_of_fraction,
                            exclude_from_resampling_with_error):
    """Generate a large set of parameter sets for later simulations."""
    generator = ParameterGenerator(balanced_result_path, distribution_path, model_path,
                                   sampling_distribution_with, exclude_from_resampling_of_fraction,
                                   exclude_from_resampling_with_error, augment=True)

    sets = generator.samples(fractions, fraction_resamples, distribution_resamples)
    parameter_ids = generator.balanced_reference.columns
    reference = generator.balanced_reference.median

    np.savez_compressed(str(to_path), fractions=fractions, parameters=parameter_ids,
                        sets=sets, reference_parameters=reference)
    return to_path


@check_for_refresh(check, metadata_path, ignore=['to_path', 'parallel', 'trajectory_dir'],
                   silent=silent)
def run_parameter_sets(model_path, parameter_sets_path, to_path,
                       trajectory_dir, trajectories_sampling_interval, step_time,
                       total_time, pulse_time, pulse_species, pulse_concentration,
                       relative_tolerance, absolute_tolerance, min_convergence_samples,
                       min_convergence_factor, max_bad, max_bad_ratio, max_outlier,
                       ignore_reactions, parallel=True):
    """Simulate and score a large set of parameter samples."""
    if trajectory_dir:
        trajectory_dir = str(trajectory_dir)

    S = simulate.RoadRunnerSimulator(str(model_path),
                                     step_time, total_time, pulse_time,
                                     pulse_species, pulse_concentration,
                                     relative_tolerance, absolute_tolerance,
                                     min_convergence_samples, min_convergence_factor,
                                     max_bad, max_bad_ratio, max_outlier,
                                     trajectory_dir, trajectories_sampling_interval,
                                     ignore_reactions)
    if parallel > 1:
        partial_results_dir = to_path.parent.joinpath('_temp_results')
        partial_results_dir.mkdir(exist_ok=True)
        scores = S.run_parallel_from_file(str(parameter_sets_path), str(partial_results_dir),
                                          parallel)
    else:
        scores = S.run_parameter_set_from_file(str(parameter_sets_path))
    np.save(str(to_path), scores)
    return to_path


@check_for_refresh(check, metadata_path, ignore=['to_path', 'parallel', 'trajectory_dir'],
                   silent=silent)
def run_parameter_sets_from_generator(model_path, balanced_result_path, distribution_path,
                                      to_path, sampling_distribution_with,
                                      exclude_from_resampling_of_fraction,
                                      exclude_from_resampling_with_error,
                                      fractions, fraction_resamples, distribution_resamples,
                                      trajectory_dir, trajectories_sampling_interval,
                                      step_time, total_time, pulse_time,
                                      pulse_species, pulse_concentration,
                                      relative_tolerance, absolute_tolerance,
                                      min_convergence_samples, min_convergence_factor,
                                      max_bad, max_bad_ratio, max_outlier,
                                      ignore_reactions, parallel=True):
    """Simulate and score a large set of parameter samples.

    Generates parameter sets on demand, which can be useful for longer runs, as the generation
    step might take to much time and disk space.
    Does NOT save the parameter sets, only the results!"""
    generator = ParameterGenerator(balanced_result_path, distribution_path, model_path,
                                   sampling_distribution_with, exclude_from_resampling_of_fraction,
                                   exclude_from_resampling_with_error, augment=True)

    if trajectory_dir:
        trajectory_dir = str(trajectory_dir)

    S = simulate.RoadRunnerSimulator(str(model_path),
                                     step_time, total_time, pulse_time,
                                     pulse_species, pulse_concentration,
                                     relative_tolerance, absolute_tolerance,
                                     min_convergence_samples, min_convergence_factor,
                                     max_bad, max_bad_ratio, max_outlier,
                                     trajectory_dir, trajectories_sampling_interval,
                                     ignore_reactions)
    if parallel > 1:
        partial_results_dir = to_path.parent.joinpath('_temp_results')
        partial_results_dir.mkdir(exist_ok=True)
        scores = S.run_parallel_from_generator(generator, fractions, fraction_resamples,
                                               distribution_resamples, str(partial_results_dir),
                                               parallel)
    else:
        scores = S.run_from_generator(generator, fractions,
                                      fraction_resamples, distribution_resamples)
    np.save(str(to_path), scores)
    return to_path


@check_for_refresh(check, metadata_path, ignore=['to_path', 'parallel', 'trajectory_dir'],
                   silent=silent)
def run_parameter_sets_from_generator_Direct_Input(model_path, balanced_result_path,
                                                   data_input_path, distribution_path, to_path,
                                                   sampling_distribution_with,
                                                   exclude_from_resampling_of_fraction,
                                                   exclude_from_resampling_with_error,
                                                   fractions, fraction_resamples,
                                                   distribution_resamples,
                                                   trajectory_dir, trajectories_sampling_interval,
                                                   step_time, total_time, pulse_time,
                                                   pulse_species, pulse_concentration,
                                                   relative_tolerance, absolute_tolerance,
                                                   min_convergence_samples, min_convergence_factor,
                                                   max_bad, max_bad_ratio, max_outlier,
                                                   ignore_reactions, parallel=True):
    """Simulate and score a large set of parameter samples using the Direct Input Generator.

    Generates parameter sets on demand, which can be useful for longer runs, as the generation
    step might take to much time and disk space.
    Does NOT save the parameter sets, only the results!"""

    generator = ParameterGenerator_Direct_Input(data_input_path, distribution_path, model_path,
                                                balanced_result_path,
                                                sampling_distribution_with,
                                                exclude_from_resampling_of_fraction,
                                                exclude_from_resampling_with_error,
                                                augment=True, stderror=0.1)

    if trajectory_dir:
        trajectory_dir = str(trajectory_dir)

    S = simulate.RoadRunnerSimulator(str(model_path),
                                     step_time, total_time, pulse_time,
                                     pulse_species, pulse_concentration,
                                     relative_tolerance, absolute_tolerance,
                                     min_convergence_samples, min_convergence_factor,
                                     max_bad, max_bad_ratio, max_outlier,
                                     trajectory_dir, trajectories_sampling_interval,
                                     ignore_reactions)
    if parallel > 1:
        partial_results_dir = to_path.parent.joinpath('_temp_results')
        partial_results_dir.mkdir(exist_ok=True)
        scores = S.run_parallel_from_generator(generator, fractions, fraction_resamples,
                                               distribution_resamples, str(partial_results_dir),
                                               parallel)
    else:
        scores = S.run_from_generator(generator, fractions,
                                      fraction_resamples, distribution_resamples)
    np.save(str(to_path), scores)
    return to_path


def run_random_model(trajectories=False):
    """Run the random model tests."""
    # Set a random seed if wanted
    seed = 1
    # Separate seed for model generation.
    # For reproducibility it is sometimes useful to keep the model / base parameters
    # but we often don't need reproducible single simulations so we use a separate seed for this.
    model_seed = 1
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # ---------------------------------------
    # Step 1: Model (Copy (A) or generate (B))
    # ---------------------------------------
    model_path = working_directory_path.joinpath('model.xml')
    #   - (B) Generate a model
    #       -> Save sbml file
    n_metabolites = 10
    n_reactions = 8
    n_compartments = 2
    C_volumes = [2, 1]
    # n_compartments = 1
    # C_volumes = [1]
    n_regulators = 4
    # n_regulators = 0
    max_degree = 8
    gamma = 1.5
    regulatory_types = [20, 462]  # Simple inhibition/activation
    # regulatory_types = [537, 636]  # Complete allosteric inhibition/activation
    # regulatory_types = [206, 533]  # Specific inhibition/activation
    _ = generate_model(model_path, n_metabolites, n_reactions, n_compartments, n_regulators,
                       max_degree, gamma, regulatory_types, C_volumes, random_seed=model_seed)

    # ---------------------------------------
    # Step 2: Parameters (Find (A) or Generate (B))
    # ---------------------------------------
    #   - Define prior distributions
    #       -> Save distributions
    # These values are already in log scale - so (0.37 ~ [1.0] ~ 2.72 for 1 sd around 1)
    distributions = {
                     'c': ['LogNormal', 0.1, 2.0],
                     'kv': ['LogNormal', 1.0, 2.0],
                     'u': ['LogNormal', 1.0, 1.6],
                     'km': ['LogNormal', 0.1, 1.2],
                     'ki': ['LogNormal', 0.1, 2.0],
                     'ka': ['LogNormal', 0.1, 2.0],
                     'keq': ['LogNormal', 1.0, 2.0],
                     'kcat': ['LogNormal', 1.0, 2.0],
                     'kcat-': ['LogNormal', 1.0, 2.0],
                     'vmax': ['LogNormal', 1.0, 2.0],
                     'mu': ['Normal', -10.0, 1.0],
                     'mu*': ['Normal', -10.0, 1.0],
                     'A': ['Normal', 0.0, 2.0],
                    }

    distribution_path = working_directory_path.joinpath('distributions.txt')
    _ = save_distributions(distribution_path, distributions)

    #   - Generate parameters
    #       -> Save generated parameters
    noise = .1
    parameter_path = working_directory_path
    true_parameter_path, noise_parameter_path = generate_parameters(model_path, distribution_path,
                                                                    parameter_path, noise,
                                                                    random_seed=model_seed)
    # Noise or not?
    parameter_path = noise_parameter_path

    #   - Balance
    #       -> Save balanced parameters
    # Reactions that should be ignored for balancing or generating kinetics.
    ignore_reactions_balancing = []
    balanced_parameter_path = working_directory_path.joinpath('parameters_balanced')
    augment = True
    sd_filler = 0.01
    balanced_result_path, balanced_parameter_path = balance_parameters(model_path, parameter_path,
                                                                       distribution_path,
                                                                       balanced_parameter_path,
                                                                       ignore_reactions_balancing,
                                                                       augment, sd_filler)

    # reaction identifier string: [(modifier identifier, [parameters], type, subtype)]
    # where parameters => [w, ki / ka] / [w, ki / ka, b]
    #           where w => multiplicity of interaction (usually 1)
    #                 ki / ka => interaction constant
    #                 b => partial factor for partial inhibition/activation
    #       type => 'inhibition' / 'activation'
    #       subtype = 'simple' / 'specific' / 'complete' / 'partial'
    # This overrides the values we previously established when creating the random parameters.
    regulation = generate_regulation(model_path, distributions, random_seed=model_seed)
    rate_law = 'CM'
    version = 'cat'
    cooperativities = None
    ignore_reactions_kinetics = []
    model_path = generate_kinetics(model_path, balanced_parameter_path, ignore_reactions_kinetics,
                                   regulation, cooperativities, rate_law, version)

    # ---------------------------------------
    # Step 3: Generate parameter sets for simulations
    # ---------------------------------------
    # We can also not generate and save the parameter sets beforehand, but create them as needed.
    # This is useful as it can be quite a large amount of data.
    save_parameters_sets = False
    # -> Save parameter sets
    if not trajectories:
        fractions = [1.00, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        fraction_resamples = 10000
        distribution_resamples = 10000
    else:
        fractions = [0.9]
        fraction_resamples = 100
        distribution_resamples = 1

    sampling_distribution_with = 1
    exclude_from_resampling_of_fraction = ('c')
    exclude_from_resampling_with_error = ('c')
    if save_parameters_sets:
        parameter_sets_path = working_directory_path.joinpath('parameter_sets.npz')
        _ = generate_parameter_sets(balanced_result_path, distribution_path, model_path,
                                    parameter_sets_path, fractions,
                                    fraction_resamples, distribution_resamples,
                                    sampling_distribution_with,
                                    exclude_from_resampling_of_fraction,
                                    exclude_from_resampling_with_error)

    # ---------------------------------------
    # Step 4: Run and score simulations
    # ---------------------------------------
    # -> Save results
    # For saving trajectories it might be interesting to not run until convergence
    # but just a standard number of times so always we have similar number of runs.
    # This can be done by making sure distribution_resamples < min_convergence_samples
    if trajectories:
        save_trajectories = working_directory_path.joinpath('trajectories')
        save_trajectories.mkdir(parents=True, exist_ok=True)
    else:
        save_trajectories = False
    trajectories_sampling_interval = 10  # Save every 10th sample.

    step_time = 0.1
    total_time = 400
    pulse_time = 200
    pulse_species = sorted(sbmlwrap.Model(str(model_path)).species.keys())[0]
    pulse_concentration = 3.5

    relative_tolerance = 1e-5
    absolute_tolerance = 1e-15

    min_convergence_samples = 1
    min_convergence_factor = 0.05
    max_bad = 250
    max_bad_ratio = 0.33
    max_outlier = 1e6

    results_data_path = working_directory_path.joinpath('results.npy')
    # False or 0 or 1 for serial
    # >= 2 for number of parallel cores
    parallel = 1

    if save_parameters_sets:
        _ = run_parameter_sets(model_path, parameter_sets_path, results_data_path,
                               save_trajectories, trajectories_sampling_interval,
                               step_time, total_time, pulse_time,
                               pulse_species, pulse_concentration,
                               relative_tolerance, absolute_tolerance,
                               min_convergence_samples, min_convergence_factor,
                               max_bad, max_bad_ratio, max_outlier, ignore_reactions_kinetics,
                               parallel)
    else:
        _ = run_parameter_sets_from_generator(model_path, balanced_result_path, distribution_path,
                                              results_data_path, sampling_distribution_with,
                                              exclude_from_resampling_of_fraction,
                                              exclude_from_resampling_with_error,
                                              fractions, fraction_resamples, distribution_resamples,
                                              save_trajectories, trajectories_sampling_interval,
                                              step_time, total_time, pulse_time,
                                              pulse_species, pulse_concentration,
                                              relative_tolerance, absolute_tolerance,
                                              min_convergence_samples, min_convergence_factor,
                                              max_bad, max_bad_ratio, max_outlier,
                                              ignore_reactions_kinetics, parallel)


def run_costa_model(trajectories=False, original_data_samples=False):
    """Run the costa model tests."""
    seed = None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Step 1: Copy Model
    model_path = working_directory_path.joinpath('model.xml')
    input_model_path = pathlib.Path('./models/Costa2014_annotated.xml')
    _ = copy_model(input_model_path, model_path)

    # Step 2: Find Parameters
    distributions = {
                     # Base quantities
                     'mu': ['Normal', -880.0, 680.00],
                     'kv': ['LogNormal', 10.0, 2.00],
                     'km': ['LogNormal', 0.01, 1.20],
                     'c':  ['LogNormal', 0.1, 2.00],
                     'u':  ['LogNormal', 1.0, 1.60],
                     'ki': ['LogNormal', 0.1, 2.00],
                     'ka': ['LogNormal', 0.1, 2.00],
                     # Derived quantities
                     'keq':   ['LogNormal', 1.0, 2.00],
                     'kcat':  ['LogNormal', 1.0, 2.00],
                     'kcat-': ['LogNormal', 1.0, 2.00],
                     'vmax':  ['LogNormal', 10.0, 2.00],
                     'A':     ['Normal', 0.0, 10.00],
                     'mu*':   ['Normal', -880.0, 680.00],
                    }
    # These values are actually in linear scaling, so convert. (median / sd)
    for key in distributions:
        if key not in {'A', 'mu', 'mu*'}:
            name, mu, sigma = distributions[key]
            distributions[key] = [name, np.log(mu), np.log(sigma)]

    distribution_path = working_directory_path.joinpath('distributions.txt')
    _ = save_distributions(distribution_path, distributions)

    organism = "Lactococcus lactis"
    # Add the additional real parameters extracted from the Costa Model.
    additional_parameters = pathlib.Path('./models/costa_parameters.csv')
    # Use existing concentrations with high certainty for balancing.
    current_concentration_error = 1e-6
    parameter_path, parameter_path_pipeline = find_parameters(model_path, organism,
                                                              working_directory_path,
                                                              additional_parameters,
                                                              current_concentration_error)

    # NOTE: Use this for a test run with no parameters.
    # Get header
    # with open(str(parameter_path), 'r') as f:
    #     header = f.readline()
    # # Wipe everything and replace header
    # with open(str(parameter_path), 'w') as f:
    #     f.write(header)

    # Step 3a: Balance the parameters

    # Define reactions that should be ignored for balancing or generating kinetics.
    ignore_reactions_balancing = [
                                  ]
    # In the Costa case  we do not ignore reactions for the balancing, just for the simulation.

    balanced_parameter_path = working_directory_path.joinpath('parameters_balanced')
    augment = True
    sd_filler = 0.01
    # Use this to limit parameters. Note that this is in short notation of parameter names
    parameter_limits = {}  # {'keq': [1e-3, 1e3]}

    balanced_result_path, balanced_parameter_path = balance_parameters(model_path, parameter_path,
                                                                       distribution_path,
                                                                       balanced_parameter_path,
                                                                       ignore_reactions_balancing,
                                                                       augment, sd_filler,
                                                                       parameter_limits)
    # Step 3b: Define regulation and create kinetics
    # Format:
    # reaction identifier string: [(modifier identifier, [parameters], type, subtype)]
    # where parameters => [w, ki / ka] / [w, ki / ka, b]
    #           where w => multiplicity of interaction (usually 1)
    #                 ki / ka => interaction constant
    #                 b => partial factor for partial inhibition/activation
    #       type => 'inhibition' / 'activation'
    #       subtype = 'simple' / 'specific' / 'complete' / 'partial'
    regulation = {
                 're12': [['FBP', [1, 1.17], 'inhibition', 'simple'],  # PTS_Glc
                          ['Pint', [1, 0.071], 'activation', 'simple']],
                 're21': [['FBP', [1, 0.039], 'activation', 'simple'],  # PYK
                          ['Pint', [1, 3.70], 'inhibition', 'simple']],
                 're22': [['FBP', [1, 0.018], 'activation', 'simple'],  # LDH
                          ['Pint', [1, 0.068], 'inhibition', 'simple']],
                 're28': [['F6P', [1, 22.03], 'inhibition', 'simple']],  # MPD
                 're24': [['ATP', [1, 6.28], 'inhibition', 'simple']],  # AE
                 're14': [['Pint', [1, 0.56], 'inhibition', 'simple']],  # P_transp
                 }
    rate_law = 'CM'
    version = 'cat'
    cooperativities = None
    ignore_reactions_kinetics = [
                                 # 're13',  # ATPase, uses as custom hill function.
                                 # 're12',  # PTS_Glc
                                 # 're13',  # ATPase
                                 # 're14',  # P_transp
                                 # # 're15',  # PGI (I - test)
                                 # 're16',  # PFK
                                 # # 're17',  # FBA (I - test)
                                 # 're18',  # GAPDH
                                 # 're20',  # ENO
                                 # 're21',  # PYK
                                 # 're22',  # LDH
                                 # # 're23',  # PDH (I - test)
                                 # # 're24',  # AE (I - test)
                                 # 're25',  # AC
                                 # 're26',  # PA
                                 # 're27',  # AB
                                 # 're28',  # MPD
                                 # 're29',  # MP
                                 # 're30',  # PTS_Man
                                 # 're31',  # Acetoin_transp
                                 # 're32',  # Mannitol_transp
                                 # 're33',  # FBPase
                                ]
    ignore_concentrations = False
    model_path = generate_kinetics(model_path, balanced_parameter_path, ignore_reactions_kinetics,
                                   regulation, cooperativities, rate_law, version,
                                   ignore_concentrations)
    # Step 4: Run and score simulations

    # Define resampling properties before starting the runs.
    # We can also not generate and save the parameter sets beforehand, but create them as needed.
    # This is useful as it can be quite a large amount of data.
    save_parameters_sets = False
    # -> Save parameter sets
    if not trajectories:
        fractions = [1.0, 0.95, 0.9, 0.6, 0.3]
        # fractions = [1.00, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        fraction_resamples = 300
        distribution_resamples = 1000
    else:
        fractions = [1.00, .9, .8, .75, .5]
        fraction_resamples = 100
        distribution_resamples = 10

    sampling_distribution_with = 5
    exclude_from_resampling_of_fraction = ('c')
    exclude_from_resampling_with_error = ('c')
    if save_parameters_sets:
        parameter_sets_path = working_directory_path.joinpath('parameter_sets.npz')
        _ = generate_parameter_sets(balanced_result_path, distribution_path, model_path,
                                    parameter_sets_path, fractions,
                                    fraction_resamples, distribution_resamples,
                                    sampling_distribution_with,
                                    exclude_from_resampling_of_fraction,
                                    exclude_from_resampling_with_error)

    # For saving trajectories it might be interesting to not run until convergence
    # but just a standard number of times so always we have similar number of runs.
    # This can be done by making sure distribution_resamples < min_convergence_samples

    if trajectories:
        save_trajectories = working_directory_path.joinpath('trajectories')
        save_trajectories.mkdir(parents=True, exist_ok=True)
    else:
        save_trajectories = False
    trajectories_sampling_interval = 10  # Save every 10th sample.

    step_time = 0.1
    total_time = 400
    pulse_time = 200
    pulse_species = 'G3P'
    pulse_concentration = 2.5

    relative_tolerance = 1e-5
    absolute_tolerance = 1e-15

    if not trajectories:
        min_convergence_samples = 25
    else:
        min_convergence_samples = distribution_resamples
    min_convergence_factor = 0.05
    max_bad = 250
    max_bad_ratio = 0.33
    max_outlier = 1e6

    results_data_path = working_directory_path.joinpath('results.npy')
    # False or 0 or 1 for serial
    # >= 2 for number of parallel cores
    parallel = 1

    if save_parameters_sets:
        if original_data_samples:
            raise NotImplementedError
        else:
            _ = run_parameter_sets(model_path, parameter_sets_path, results_data_path,
                                   save_trajectories, trajectories_sampling_interval,
                                   step_time, total_time, pulse_time,
                                   pulse_species, pulse_concentration,
                                   relative_tolerance, absolute_tolerance,
                                   min_convergence_samples, min_convergence_factor,
                                   max_bad, max_bad_ratio, max_outlier, ignore_reactions_kinetics,
                                   parallel)

    else:
        if original_data_samples:
            _ = run_parameter_sets_from_generator_Direct_Input(model_path, balanced_result_path,
                                                               parameter_path, distribution_path,
                                                               results_data_path,
                                                               sampling_distribution_with,
                                                               exclude_from_resampling_of_fraction,
                                                               exclude_from_resampling_with_error,
                                                               fractions, fraction_resamples,
                                                               distribution_resamples,
                                                               save_trajectories,
                                                               trajectories_sampling_interval,
                                                               step_time, total_time, pulse_time,
                                                               pulse_species, pulse_concentration,
                                                               relative_tolerance,
                                                               absolute_tolerance,
                                                               min_convergence_samples,
                                                               min_convergence_factor,
                                                               max_bad, max_bad_ratio, max_outlier,
                                                               ignore_reactions_kinetics,
                                                               parallel)
        else:
            _ = run_parameter_sets_from_generator(model_path, balanced_result_path,
                                                  distribution_path,
                                                  results_data_path, sampling_distribution_with,
                                                  exclude_from_resampling_of_fraction,
                                                  exclude_from_resampling_with_error,
                                                  fractions, fraction_resamples,
                                                  distribution_resamples,
                                                  save_trajectories,
                                                  trajectories_sampling_interval,
                                                  step_time, total_time, pulse_time,
                                                  pulse_species, pulse_concentration,
                                                  relative_tolerance, absolute_tolerance,
                                                  min_convergence_samples, min_convergence_factor,
                                                  max_bad, max_bad_ratio, max_outlier,
                                                  ignore_reactions_kinetics, parallel)


def run_pipeline_only(path, organism, balance):
    """Run the pipeline only tests."""
    seed = 1

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Step 1: Copy Model
    model_path = working_directory_path.joinpath('model.xml')
    input_model_path = pathlib.Path(path)
    _ = copy_model(input_model_path, model_path)

    # Step 2: Find Parameters
    distributions = {
                     # Base quantities
                     'mu': ['Normal', -880.0, 680.00],
                     'kv': ['LogNormal', 10.0, 2.00],
                     'km': ['LogNormal', 0.01, 1.20],
                     'c':  ['LogNormal', 0.1, 2.00],
                     'u':  ['LogNormal', 1.0, 1.60],
                     'ki': ['LogNormal', 0.1, 2.00],
                     'ka': ['LogNormal', 0.1, 2.00],
                     # Derived quantities
                     'keq':   ['LogNormal', 1.0, 2.00],
                     'kcat':  ['LogNormal', 1.0, 2.00],
                     'kcat-': ['LogNormal', 1.0, 2.00],
                     'vmax':  ['LogNormal', 10.0, 2.00],
                     'A':     ['Normal', 0.0, 10.00],
                     'mu*':   ['Normal', -880.0, 680.00],
                    }
    # These values are actually in linear scaling, so convert. (median / sd)
    for key in distributions:
        if key not in {'A', 'mu', 'mu*'}:
            name, mu, sigma = distributions[key]
            distributions[key] = [name, np.log(mu), np.log(sigma)]

    distribution_path = working_directory_path.joinpath('distributions.txt')
    _ = save_distributions(distribution_path, distributions)

    parameter_path, parameter_path_pipeline = find_parameters(model_path, organism,
                                                              working_directory_path)
    if balance:
        # Step 3a: Balance the parameters
        balanced_parameter_path = working_directory_path.joinpath('parameters_balanced')
        augment = True
        sd_filler = 0.01
        # parameter_limits = {'keq': [1e-4, 1e4],
                            # 'kcat': [1e-6, 1e3]}
        parameter_limits = {}
        balanced_result_path, balanced_parameter_path = balance_parameters(model_path,
                                                                           parameter_path,
                                                                           distribution_path,
                                                                           balanced_parameter_path,
                                                                           [], augment,
                                                                           sd_filler,
                                                                           parameter_limits)

        regulation = {
                     }
        rate_law = 'CM'
        version = 'cat'
        cooperativities = None
        ignore_reactions_kinetics = []
        ignore_concentrations = False
        model_path = generate_kinetics(model_path, balanced_parameter_path,
                                       ignore_reactions_kinetics,
                                       regulation, cooperativities, rate_law, version,
                                       ignore_concentrations)


def run_colicore_model(version='core'):
    """Run the E. coli core model tests."""
    seed = 1

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Step 1: Copy Model
    model_path = working_directory_path.joinpath('model.xml')
    if version == 'core':
        input_model_path = pathlib.Path('./models/ecoli_core_model.xml')
    elif version == 'pruned':
        input_model_path = pathlib.Path('./models/coli_core_pruned.xml')
    elif version == 'reduced':
        input_model_path = pathlib.Path('./models/coli_core_compressed.xml')
    else:
        raise ValueError("Invalid version {core / pruned / reduced}!")
    _ = copy_model(input_model_path, model_path)

    # Step 2: Find Parameters
    distributions = {
                     # Base quantities
                     'mu': ['Normal', -880.0, 4.00],
                     'kv': ['LogNormal', 10.0, 2.00],
                     'km': ['LogNormal', 0.01, 1.20],
                     'c':  ['LogNormal', 0.1, 2.00],
                     'u':  ['LogNormal', 1.0, 1.60],
                     'ki': ['LogNormal', 0.1, 2.00],
                     'ka': ['LogNormal', 0.1, 2.00],
                     # Derived quantities
                     'keq':   ['LogNormal', 1.0, 2.00],
                     'kcat':  ['LogNormal', 1.0, 2.00],
                     'kcat-': ['LogNormal', 1.0, 2.00],
                     'vmax':  ['LogNormal', 10.0, 2.00],
                     'A':     ['Normal', 0.0, 10.00],
                     'mu*':   ['Normal', -880.0, 680.00],
                    }
    # These values are actually in linear scaling, so convert. (median / sd)
    for key in distributions:
        if key not in {'A', 'mu', 'mu*'}:
            name, mu, sigma = distributions[key]
            distributions[key] = [name, np.log(mu), np.log(sigma)]

    distribution_path = working_directory_path.joinpath('distributions.txt')
    _ = save_distributions(distribution_path, distributions)

    organism = "Escherichia coli"
    parameter_path, parameter_path_pipeline = find_parameters(model_path, organism,
                                                              working_directory_path)

    # Step 3a: Balance the parameters

    # Define reactions that should be ignored for balancing or generating kinetics.
    ignore_reactions_balancing = []

    balanced_parameter_path = working_directory_path.joinpath('parameters_balanced')
    augment = True
    sd_filler = 0.01
    balanced_result_path, balanced_parameter_path = balance_parameters(model_path, parameter_path,
                                                                       distribution_path,
                                                                       balanced_parameter_path,
                                                                       ignore_reactions_balancing,
                                                                       augment, sd_filler)
    # Step 3b: Define regulation and create kinetics
    # Format:
    # reaction identifier string: [(modifier identifier, [parameters], type, subtype)]
    # where parameters => [w, ki / ka] / [w, ki / ka, b]
    #           where w => multiplicity of interaction (usually 1)
    #                 ki / ka => interaction constant
    #                 b => partial factor for partial inhibition/activation
    #       type => 'inhibition' / 'activation'
    #       subtype = 'simple' / 'specific' / 'complete' / 'partial'
    regulation = {}
    rate_law = 'CM'
    version = 'cat'
    cooperativities = None
    ignore_reactions_kinetics = []
    model_path = generate_kinetics(model_path, balanced_parameter_path, ignore_reactions_kinetics,
                                   regulation, cooperativities, rate_law, version)

    # Step 4: Run and score simulations

    # Define resampling properties before starting the runs.
    # We can also not generate and save the parameter sets beforehand, but create them as needed.
    # This is useful as it can be quite a large amount of data.
    save_parameters_sets = False
    # -> Save parameter sets

    # For this test we just want 1 simulation at the mean value.
    fractions = [1.00]
    fraction_resamples = 1
    distribution_resamples = 1
    sampling_distribution_with = 0

    exclude_from_resampling_of_fraction = ('c')
    exclude_from_resampling_with_error = ('c')
    if save_parameters_sets:
        parameter_sets_path = working_directory_path.joinpath('parameter_sets.npz')
        _ = generate_parameter_sets(balanced_result_path, distribution_path, model_path,
                                    parameter_sets_path, fractions,
                                    fraction_resamples, distribution_resamples,
                                    sampling_distribution_with,
                                    exclude_from_resampling_of_fraction,
                                    exclude_from_resampling_with_error)

    # For saving trajectories it might be interesting to not run until convergence
    # but just a standard number of times so always we have similar number of runs.
    # This can be done by making sure distribution_resamples < min_convergence_samples

    save_trajectories = working_directory_path.joinpath('trajectories')
    save_trajectories.mkdir(parents=True, exist_ok=True)
    trajectories_sampling_interval = 10  # Save every 10th sample.

    step_time = 0.1
    total_time = 400
    pulse_time = 200
    pulse_species = 'M_g3p_c'
    pulse_concentration = 2.5

    relative_tolerance = 1e-5
    absolute_tolerance = 1e-15

    min_convergence_samples = 1
    min_convergence_factor = 0.05
    max_bad = 250
    max_bad_ratio = 0.33
    max_outlier = 1e6

    results_data_path = working_directory_path.joinpath('results.npy')
    # False or 0 or 1 for serial
    # >= 2 for number of parallel cores
    parallel = 16

    if save_parameters_sets:
        _ = run_parameter_sets(model_path, parameter_sets_path, results_data_path,
                               save_trajectories, trajectories_sampling_interval,
                               step_time, total_time, pulse_time,
                               pulse_species, pulse_concentration,
                               relative_tolerance, absolute_tolerance,
                               min_convergence_samples, min_convergence_factor,
                               max_bad, max_bad_ratio, max_outlier, ignore_reactions_kinetics,
                               parallel)
    else:
        _ = run_parameter_sets_from_generator(model_path, balanced_result_path, distribution_path,
                                              results_data_path, sampling_distribution_with,
                                              exclude_from_resampling_of_fraction,
                                              exclude_from_resampling_with_error,
                                              fractions, fraction_resamples, distribution_resamples,
                                              save_trajectories, trajectories_sampling_interval,
                                              step_time, total_time, pulse_time,
                                              pulse_species, pulse_concentration,
                                              relative_tolerance, absolute_tolerance,
                                              min_convergence_samples, min_convergence_factor,
                                              max_bad, max_bad_ratio, max_outlier,
                                              ignore_reactions_kinetics, parallel)


if __name__ == '__main__':
    # Multiprocessing and mkl threading don't always play nice.
    # If you're doing a run with a lot of simulations, it can be better to limit
    # everyone to a single core and instead just split the tasks over more processes.
    # For balancing a big model however, it can be significantly faster.
    no_mkl = True
    if no_mkl:
        try:
            import mkl
            mkl.set_num_threads(1)
            print("Set mkl threads to 1")
        except ImportError:
            print("No mkl-service installed. If using conda: conda install mkl-service")

    # Change here which example you want to run.
    model = 'pipeline_only'  # 'costa' / 'random' / 'pipeline_only' / 'colicore'
    trajectories = True  # True / False - not relevant for 'pipeline_only' or 'colicore'
    version = 'reduced'  # 'core' / 'pruned' / 'reduced' - only relevant for 'colicore'
    balance = True  # Only for 'pipeline_only'
    # For Costa only: Use direct input for the initial sample or balance first and sample that.
    original_data_samples = False  # For Costa only.

    # Change the output directory at the top of the script!
    working_directory = pathlib.Path('./test_data/example')
    if working_directory_path != working_directory:
        raise Exception("Make sure to update the working directory at the top of the script!")
    print("Working in {}".format(working_directory))

    # If you haven't build a cache of the online resources yet, make sure to set
    # skipbrendalogin to False, and you'll be prompted on the command line for your
    # username (email) and password

    if model == 'costa':
        # Costa et al. 2014 Lactococcus lactis model.
        # This is an existing fully kinetic model using similar rate laws.
        run_costa_model(trajectories, original_data_samples)

    elif model == 'random':
        # A randomly generated reaction network kineticized with random parameters.
        run_random_model(trajectories)

    elif model == 'pipeline_only':
        # Note: Very large models without kinetics. Integration impossible because
        # of system size / speed differences.
        # Balancing is possible but requires a large amount of RAM. (>40 gb)

        # Orth et al 2011 Escherichia coli genome scale metabolic model.
        # path = pathlib.Path('./models/iJO1366.xml')
        # path = pathlib.Path('./models/iJO1366_compressed.xml')
        # path = pathlib.Path('./models/iJO1366_pruned.xml')
        # organism = "Escherichia coli"
        # run_pipeline_only(path, organism, balance)

        # # Osterlund et al 2013  S. cerevisiae genome scale metabolic model.
        # path = pathlib.Path('./models/iTO977_v1.00_raven.xml')
        # organism = "Saccharomyces cerevisiae"
        # run_pipeline_only(path, organism, balance)

        # path = pathlib.Path('./models/ecoli_core_model.xml')
        # organism = "Escherichia coli"
        # run_pipeline_only(path, organism, balance)

        path = pathlib.Path('./models/Costa2014_annotated.xml')
        organism = "Lactococcus lactis"
        run_pipeline_only(path, organism, balance)

    elif model == 'colicore':
        # Smaller  Escherichia coli genome scale metabolic model.
        # EcoSal Chapter 10.2.1 - Reconstruction and Use of Microbial Metabolic Networks:
        # the Core Escherichia coli Metabolic Model as an Educational Guide
        # by Orth, Fleming, and Palsson (2010)
        # See also: http://gcrg.ucsd.edu/Downloads/EcoliCore
        run_colicore_model(version)
