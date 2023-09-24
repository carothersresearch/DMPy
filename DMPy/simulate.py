#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tools to simulate SBML models.

Author: Rik van Rosmalen
"""
# Python 2/3 compatibility
from __future__ import print_function
from __future__ import division

# Standard library
import os
import sys
import contextlib
import multiprocessing
import warnings
import random

# External modules
import numpy as np
import scipy.stats
import roadrunner as rr


# Utility functions
def _fileno(file_or_fd):
    """Get the filenumber of a file or file descriptor."""
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def _stderr_redirected(to=os.devnull, stdout=None):
    """
    Context manager to temporarily suppress stderr.

    See: http://stackoverflow.com/a/22434262/190597 (J.F. Sebastian).
    """
    if stdout is None:
        stdout = sys.stderr

    stdout_fd = _fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(_fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


@contextlib.contextmanager
def _dummy_context_mgr():
    """Dummy context manager that does nothing."""
    yield None


class Simulator(object):
    """Base simulator object. Partial interface left to be implemented by subclasses.

    The methods that should be implemented in a child class are the following:
        '__init__'             - Set up simulation specifics. Save all arguments in self._args for
                                 recreating the object in parallel mode.
        'set_parameters'       - Override the model parameters with a new set.
                                 parameters: {[type, metabolite, reaction]: value}
        'run'                  - Run the model, return (t, y) as numpy arrays
        'get_metabolite_index' - Get the metabolite index for the trajectories.

    TODO: Clean up mess with different version for excisting parameter set vs parameter generator.
    """

    def __init__(self, sbml_model,
                 step_time, total_time, pulse_time,
                 pulse_species, pulse_concentration,
                 relative_tolerance, absolute_tolerance,
                 min_convergence_samples, min_convergence_factor,
                 max_bad, max_bad_ratio, max_outlier,
                 save_trajectories, trajectories_sampling_interval,
                 silence_warnings=True):
        """Create an object for simulation."""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Change the parameters of the underlying model."""
        raise NotImplementedError

    def set_reference(self, parameters):
        """Set the parameters and run a reference simulation for scoring."""
        self.reference_parameters = parameters
        self.set_parameters(parameters)
        self.reference_t, self.reference_y = self.run()

    def get_metabolite_index(self):
        """Get the index of the metabolite id into the trajectory results."""
        raise NotImplementedError

    def run(self):
        """Run a single simulation run with the current parameters."""
        raise NotImplementedError

    def set_trajectory_save_settings(self, buffer_size=100 * 1024 ** 2):
        """Set the required settings for the saving the trajectories."""
        self._trajectory_buffer_size = buffer_size
        self._trajectory_buffer = None
        self._trajectory_buffer_index = 0
        self._trajectory_buffer_iteration = 0
        self._trajectory_meta = []

    def save_trajectory(self, fraction, fs, ds, trajectory):
        """Save a trajectory. Utilize a buffer to reduce write slowdown."""
        if self._trajectory_buffer is None:
            max_size = int(self._trajectory_buffer_size / trajectory.nbytes)
            self._trajectory_buffer = np.zeros((max_size,) + trajectory.shape)

        self._trajectory_buffer[self._trajectory_buffer_index, :] = trajectory
        self._trajectory_buffer_index += 1
        self._trajectory_meta.append((fraction, fs, ds))

        # Check if it's full and save.
        if self._trajectory_buffer_index == self._trajectory_buffer.shape[0]:
            self.write_trajectory_buffer()

            # Clear out all buffer (settings)
            self._trajectory_buffer.fill(0)
            self._trajectory_buffer_index = 0
            self._trajectory_meta = []

            self._trajectory_buffer_iteration += 1

    def write_trajectory_buffer(self):
        """Write the everything in the current trajectory buffer to disk.

        Does not empty the buffer. Only writes if there is something in the buffer."""
        # Use the current process id to make sure that we have an unique file when
        # running in parallel.
        if self._trajectory_buffer_index > 0:
            path = (self.trajectory_dir + '/' + str(id(multiprocessing.current_process())) +
                    '_' + str(self._trajectory_buffer_iteration) + '.npz')
            # Only save the populated part of the buffer.
            trajectory = self._trajectory_buffer[:self._trajectory_buffer_index]
            np.savez(path, trajectory=trajectory, run=np.array(self._trajectory_meta),
                     reference=self.reference_y, time=self.reference_t,
                     index=self.get_metabolite_index())

    def score(self, t, y):
        """Score the trajectory against the reference trajectory.

        If t does not match the reference t, but the starting and ending time match,
        y will be linearly interpolated."""
        if not np.all(t == self.reference_t):
            if t[0] == self.reference_t[0] and t[-1] == self.reference_t[-1]:
                # Interpolate y to match reference
                y = np.interp(self.reference_t, t, y)
            else:
                raise ValueError("Start and end times don't match reference.")
        # Mean square error
        return np.square(self.reference_y - y).mean()

    def load_parameter_set(self, parameter_set_path):
        """Load parameter sets from path.

        The parameter set file should be a numpy archive consisting of:
            'sets': [fractions, fraction_sample, distribution sample, parameters]
            'parameters': [parameter keys]
            'fractions': [fraction keys]
        """
        x = np.load(parameter_set_path)
        sets = x['sets']
        parameters = [tuple(i) for i in x['parameters']]
        fractions = x['fractions']
        reference_parameters = x['reference_parameters']
        return sets, parameters, fractions, reference_parameters

    def run_parameter_set(self, sets, parameters, fractions):
        """Run parameter sets and return scores."""
        if self.trajectory_dir:
            self.set_trajectory_save_settings()

        core = id(multiprocessing.current_process())

        scores = []
        for fi in range(sets.shape[0]):
            for fs in range(sets.shape[1]):
                bad = 0
                good = 0
                run_scores = []
                for ds in range(sets.shape[2]):
                    parameter_values = sets[fi, fs, ds]
                    self.set_parameters(dict(zip(parameters, parameter_values)))
                    try:
                        t, y = self.run()
                    except RuntimeError:
                        score = np.NaN
                        bad += 1
                    else:
                        score = self.score(t, y)
                        if score > self.max_outlier:
                            score = np.NaN
                            bad += 1
                        else:
                            good += 1
                        # Do we need to save the trajectory?
                        if self.trajectory_dir:
                            self.save_trajectory(fractions[fi], fs, ds,
                                                 y[::self.trajectories_sampling_interval])

                    run_scores.append(score)
                    scores.append((fractions[fi], fs, core, ds, score))
                    # Keep track of stats we need to check for continuation.
                    if bad:
                        bad_ratio = bad / (good + bad)
                    else:
                        bad_ratio = 0

                    # If all run_scores are nan, this will give a warning we don't care about.
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        convergence_factor = (scipy.stats.sem(run_scores, nan_policy='omit') /
                                              np.nanmean(run_scores))
                        # print(convergence_factor, good, bad, score)
                    # Do we need to quit this run yet?
                    done = False
                    if ds >= self.min_convergence_samples - 1:
                        if bad >= self.max_bad:
                            # print("Max bad obtained {} >= {}".format(bad, self.max_bad))
                            done = True
                        elif bad_ratio >= self.max_bad_ratio:
                            # print("Max bad ratio obtained {} >= {}".format(bad_ratio,
                            #                                               self.max_bad_ratio))
                            done = True
                        elif convergence_factor <= self.min_convergence_factor:
                            # print("Convergence obtained {} <= {}"
                            #      "".format(convergence_factor, self.min_convergence_factor))
                            done = True

                    if done:
                        break
                # else:
                #    print("Max parameter sets achieved {} - {}".format(ds, convergence_factor))

        # Make sure to flush the buffer to disk for the last values.
        if self.trajectory_dir:
            self.write_trajectory_buffer()

        return scores

    def run_from_generator(self, generator, fractions,
                           fraction_resamples, distribution_resamples):
        """Run parameter sets and return scores."""
        parameters = [tuple(i) for i in generator.balanced_reference.columns]

        # NOTE: The extra balancing is only implemneted when running from generator.
        # TODO: Make this a setting, merge with running form excisting parameter sets.
        # reference = generator.balanced_reference.median
        # reference_is_balanced_twice = True
        # if reference_is_balanced_twice:
        temp = generator.sampling_distribution_with
        try:
            generator.sampling_distribution_with = 0
            self.set_reference(dict(zip(parameters,
                                        next(generator.sample_generator(1.0)))))
        finally:
            generator.sampling_distribution_with = temp
        # else:
            # self.set_reference(dict(zip(parameters, reference)))

        if self.trajectory_dir:
            self.set_trajectory_save_settings()

        core = id(multiprocessing.current_process())

        scores = []
        for fraction in fractions:
            for fs in range(fraction_resamples):
                bad = 0
                good = 0
                run_scores = []
                for ds, parameter_values in enumerate(generator.sample_generator(fraction)):
                    self.set_parameters(dict(zip(parameters, parameter_values)))
                    try:
                        t, y = self.run()
                    except RuntimeError:
                        score = np.NaN
                        bad += 1
                    else:
                        score = self.score(t, y)
                        if score > self.max_outlier:
                            score = np.NaN
                            bad += 1
                        else:
                            good += 1
                        # Do we need to save the trajectory?
                        if self.trajectory_dir:
                            self.save_trajectory(fraction, fs, ds,
                                                 y[::self.trajectories_sampling_interval])

                    run_scores.append(score)
                    scores.append((fraction, fs, core, ds, score))
                    # Keep track of stats we need to check for continuation.
                    if bad:
                        bad_ratio = bad / (good + bad)
                    else:
                        bad_ratio = 0

                    # If all run_scores are nan, this will give a warning we don't care about.
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        convergence_factor = (scipy.stats.sem(run_scores, nan_policy='omit') /
                                              np.nanmean(run_scores))
                        # print(convergence_factor, good, bad, score)
                    # Do we need to quit this run yet?
                    done = False
                    if ds >= self.min_convergence_samples - 1:
                        if bad >= self.max_bad:
                            # print("Max bad obtained {} >= {}".format(bad, self.max_bad))
                            done = True
                        elif bad_ratio >= self.max_bad_ratio:
                            # print("Max bad ratio obtained {} >= {}".format(bad_ratio,
                            #                                                self.max_bad_ratio))
                            done = True
                        elif convergence_factor <= self.min_convergence_factor:
                            # print("Convergence obtained {} <= {}"
                            #       "".format(convergence_factor, self.min_convergence_factor))
                            done = True
                        elif ds >= distribution_resamples - 1:
                            # print("Max parameter sets achieved {} - {}".format(ds,
                            #                                                    convergence_factor))
                            done = True

                    if done:
                        break

        # Make sure to flush the buffer to disk for the last values.
        if self.trajectory_dir:
            self.write_trajectory_buffer()

        return scores

    def run_parameter_set_from_file(self, parameter_set_path):
        """Load parameter sets from path, run sequentially and return scores.

        The parameter set file should be a numpy archive consisting of:
            'sets': [fraction, fraction_sample, distribution sample, p1, p2 ...]
            'key': [column names for sets]
        """
        sets, par_names, fracts, ref_set = self.load_parameter_set(parameter_set_path)
        self.set_reference(dict(zip(par_names, ref_set)))
        return self.run_parameter_set(sets, par_names, fracts)

    def run_parallel_from_file(self, parameter_set_path, partial_results_dir, n_processes=None):
        """Load parameter sets from path, run in parallel and return scores.

        The parameter set file should be a numpy archive consisting of:
            'sets': [fraction, fraction_sample, distribution sample, p1, p2 ...]
            'key': [column names for sets]

        The default number of parallel processes is cpu_count - 1
        """
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1
        sets, parameters, fractions, ref_set = self.load_parameter_set(parameter_set_path)

        reference_parameters = dict(zip(parameters, ref_set))

        # Split work. We split over the second axis so we don't put all similar fractions
        # in the same groups, since some fraction will likely convergence faster.
        subsets = np.array_split(sets, n_processes, axis=1)

        processes = []
        for i in range(n_processes):
            pipe_in, pipe_out = multiprocessing.Pipe(duplex=False)
            args = (subsets[i], parameters, fractions, reference_parameters,
                    pipe_out, partial_results_dir, self._args)

            proc = multiprocessing.Process(target=type(self)._parallel_helper, args=args)
            proc.start()
            processes.append((proc, pipe_in))

        files = []
        for proc, pipe in processes:
            files.append(pipe.recv())
            proc.join()

        scores = []
        for file in files:
            scores.append(np.load(file))

        return np.concatenate(scores)

    def run_parallel_from_generator(self, generator, fractions, fraction_resamples,
                                    distribution_resamples, partial_results_dir, n_processes=None):
        """Load parameter sets from path, run in parallel and return scores.

        The parameter set file should be a numpy archive consisting of:
            'sets': [fraction, fraction_sample, distribution sample, p1, p2 ...]
            'key': [column names for sets]

        The default number of parallel processes is cpu_count - 1
        """
        if n_processes is None:
            n_processes = multiprocessing.cpu_count() - 1

        # Split work into as equal as possible integer parts.
        f_samples_each, one_extra = divmod(fraction_resamples, n_processes)
        f_samples = np.zeros(n_processes, dtype=int) + f_samples_each
        f_samples[:one_extra] += 1
        # f_samples.sum() == fraction_resamples

        processes = []
        for i, fraction_resamples in enumerate(f_samples):
            pipe_in, pipe_out = multiprocessing.Pipe(duplex=False)
            args = (generator, fractions, fraction_resamples, distribution_resamples,
                    pipe_out, partial_results_dir, self._args)
            proc = multiprocessing.Process(target=type(self)._parallel_helper_generator,
                                           args=args)
            proc.start()
            processes.append((proc, pipe_in))

        files = []
        for proc, pipe in processes:
            files.append(pipe.recv())
            proc.join()

        scores = []
        for file in files:
            scores.append(np.load(file))

        return np.concatenate(scores)

    @classmethod
    def _parallel_helper(cls, sets, parameters, fractions, reference_parameters, pipe_out,
                         partial_results_dir, args):
        """Helper function for use in parallel simulations."""
        simulator = cls(*args)
        simulator.set_reference(reference_parameters)
        # Set a new random state, since otherwise the state of the RNG will be
        # duplicated over multiple processes, leading to identical simulations.
        # Ideally this would use an algorith supporting multiple streams, since
        # there is no guarentee of non-overlap, but this is not easily done in numpy.
        np.random.seed(id(multiprocessing.current_process()) % (2**32 - 1))
        random.seed(id(multiprocessing.current_process()) % (2**32 - 1))

        scores = simulator.run_parameter_set(sets, parameters, fractions)

        # Save results and send filename back.
        p_id = str(id(multiprocessing.current_process()))
        filename = os.path.join(partial_results_dir, 'partial_' + p_id + '.npy')
        np.save(filename, scores)
        pipe_out.send(filename)

    @classmethod
    def _parallel_helper_generator(cls, generator, fractions, fraction_resamples,
                                   distribution_resamples, pipe_out, partial_results_dir, args):
        simulator = cls(*args)
        # Set a new random state, since otherwise the state of the RNG will be
        # duplicated over multiple processes, leading to identical simulations.
        # Ideally this would use an algorith supporting multiple streams, since
        # there is no guarentee of non-overlap, but this is not easily done in numpy.
        np.random.seed(id(multiprocessing.current_process()) % (2**32 - 1))
        random.seed(id(multiprocessing.current_process()) % (2**32 - 1))

        scores = simulator.run_from_generator(generator, fractions, fraction_resamples,
                                              distribution_resamples)
        # Save results and send filename back.
        p_id = str(id(multiprocessing.current_process()))
        filename = os.path.join(partial_results_dir, 'partial_' + p_id + '.npy')
        np.save(filename, scores)
        pipe_out.send(filename)


class RoadRunnerSimulator(Simulator):
    """Simulator using libRoadRunner for simulation."""

    def __init__(self, sbml_model,
                 step_time, total_time, pulse_time,
                 pulse_species, pulse_concentration,
                 relative_tolerance, absolute_tolerance,
                 min_convergence_samples, min_convergence_factor,
                 max_bad, max_bad_ratio, max_outlier,
                 trajectory_dir, trajectories_sampling_interval,
                 ignore_reactions, silence_warnings=True):
        """Create a libRoadRunner simulator wrapper instance for a model."""
        self.step_time = step_time
        self.total_time = total_time

        self.pulse_time = pulse_time
        self.pulse_species = pulse_species
        self.pulse_concentration = pulse_concentration

        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance
        self.silence_warnings = silence_warnings

        self.min_convergence_samples = min_convergence_samples
        self.min_convergence_factor = min_convergence_factor
        self.max_bad = max_bad
        self.max_bad_ratio = max_bad_ratio
        self.max_outlier = max_outlier

        self.trajectory_dir = trajectory_dir
        self.trajectories_sampling_interval = trajectories_sampling_interval

        # Reactions to be ignored when setting parameters
        self.ignore_reactions = ignore_reactions

        # Save arguments in case we need to replicate the object for multiprocessing purposes.
        self._args = (sbml_model,
                      step_time, total_time, pulse_time,
                      pulse_species, pulse_concentration,
                      relative_tolerance, absolute_tolerance,
                      min_convergence_samples, min_convergence_factor,
                      max_bad, max_bad_ratio, max_outlier,
                      trajectory_dir, trajectories_sampling_interval,
                      ignore_reactions, silence_warnings)

        # Setup roadrunner model an relevant settings.
        self.rr_model = rr.RoadRunner(sbml_model)

        # Convert parameters to global so we can modify them later.
        self.rr_model = rr.RoadRunner(self.rr_model.getParamPromotedSBML(self.rr_model.getSBML()))
        self.rr_model.integrator.relative_tolerance = self.relative_tolerance
        self.rr_model.integrator.absolute_tolerance = self.absolute_tolerance

        # Setup parameter map
        self.parameter_map = None

    def get_metabolite_index(self):
        """Get the index of the metabolite id into the trajectory results."""
        # Older versions of libroadrunner
        try:
            return np.array(self.rr_model.getModel().getStateVectorIds())
        # Newer versions deprecated getStateVectorIds in favor of getFloatingSpeciesIds
        except AttributeError:
            return np.array(self.rr_model.getModel().getFloatingSpeciesIds())

    def _map_parameter_names(self, parameters):
        """Create parameter names from the columns that can map to the model parameter names."""
        # Only variables we need to change
        names = {'km': 'kM',
                 'u': 'u',
                 'kcat': 'kcatprod',
                 'kcat-': 'kcatsub',
                 }
        mapping = {}
        for p_type, metabolite, reaction in parameters:
            # Set to false so we know we don't care in change_parameters
            if p_type not in names or reaction in self.ignore_reactions:
                mapped = False
            elif metabolite is not None:
                mapped = "{}_{}_{}_{}".format(reaction, names[p_type], reaction, metabolite)
            else:
                mapped = "{}_{}_{}".format(reaction, names[p_type], reaction)
            mapping[(p_type, metabolite, reaction)] = mapped
        return mapping

    def run(self):
        """Run the model with the current parameters and settings."""
        try:
            with _stderr_redirected() if self.silence_warnings else _dummy_context_mgr():
                # Simulate before pulse
                r1 = self.rr_model.simulate(0, self.pulse_time,
                                            int(self.pulse_time / self.step_time))

            # Pulse
            name = '[{}]'.format(self.pulse_species)
            self.rr_model[name] = self.pulse_concentration
            with _stderr_redirected() if self.silence_warnings else _dummy_context_mgr():
                # Simulate after pulse
                r2 = self.rr_model.simulate(self.pulse_time, self.total_time,
                                            int((self.total_time - self.pulse_time) /
                                                self.step_time))
            # Combine and return results
            x = np.vstack((r1, r2))
            return x[:, 0], x[:, 1:]

        finally:
            # Always reset model!
            self.rr_model.reset()

    def set_parameters(self, parameters):
        """Change the parameters of the model."""
        if self.parameter_map is None:
            self.parameter_map = self._map_parameter_names(parameters.keys())

        for key, value in parameters.items():
            key = self.parameter_map[key]
            if key:
                self.rr_model[key] = value
