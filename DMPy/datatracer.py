#!/usr/bin/env python
"""
Scripts to link data required and available.

Author: Rik van Rosmalen
"""
from __future__ import division
from __future__ import print_function

import logging
import collections
import multiprocessing
import time

try:
    import dill
except ImportError:
    import pickle as dill


class Tracer(object):
    """Class to find data wanted from the data sources available."""

    def __init__(self, tasks, transforms):
        """Create the DataTracer object with a set of tasks and transforms.

        Arguments:
        tasks - a list of Tasks to be done.
        transforms - a list of data transformation Transforms."""
        self.tasks = tasks
        self.graph = collections.defaultdict(list)
        self.transforms = []

        for transform in transforms:
            self.registerTransform(transform)

    def registerTransform(self, transform):
        """Register a data Transform.

        Arguments:
        transforms - a data transformation Transform object."""
        self.transforms.append(transform)
        # Update graph
        li = self.graph[transform.required]
        # Update possible destinations
        li.append(transform)
        # Make sure to keep it sorted
        li.sort(key=lambda x: getattr(x, 'weight'))
        # Update graph
        self.graph[transform.required] = li
        logging.info("TRACER: Added {}".format(str(transform)))

    def run(self):
        """Try to solve all current Tasks."""
        # TODO: Update logging
        logging.info("TRACER: Running {} tasks".format(len(self.tasks)))
        finished = []
        que = []

        logging.info("TRACER: Finding paths")
        for task in self.tasks:
            task.path = self.trace(task)
            que.append(task)

        logging.info("TRACER: Following paths")
        while que:
            task = que.pop()
            # We can quit once we've obtained our goals.
            while task.wanted:
                try:
                    transform = task.next()
                except ValueError:  # If none of the steps work any more, break out of the loop.
                    logging.info("TRACER: Could not finish task: {}".format(task))
                    break
                # Else update the task
                logging.info("TRACER: Running {}".format(transform))
                task = transform(task)
            else:  # Only runs when loop ends without break.
                logging.info("TRACER: Finished task successfully: {}".format(task))
            finished.append(task)
        self.tasks = finished

    def run_async(self, processes=1, ignore_error=False):
        """Try to solve all current Tasks.

        Tasks will be solved asynchronously using a pool of worker processes.
        Warning: Not functional atm with the single write access cache databases!!!
        """
        if not ignore_error:
            raise NotImplementedError("Not compatible with current database cache implementation.")
        logging.info("TRACER: Running {} tasks".format(len(self.tasks)))
        # Setup process, pool etc.
        pool = multiprocessing.Pool(processes)
        results = []  # list of async results
        queue = []  # list of tasks that need to be handed to the pool.
        finished = []
        logging.info("TRACER: Finding paths")
        for task in self.tasks:
            task.path = self.trace(task)
            queue.append(task)

        # Event loop.
        # 1) Post unsolved tasks with a path to pool
        # 2) Check for solved tasks.
        # 3) Break when no more queue nor results.
        logging.info("TRACER: Following paths")
        while queue or results:
            while queue:
                task = queue.pop()
                try:
                    transform = task.next()
                except ValueError:
                    finished.append(task)
                    logging.info("TRACER: Could not finish task: {}".format(task))
                    continue
                logging.info("TRACER: Added task to pool: {} {}".format(task, transform))
                # results.append(pool.apply_async(transform, task))
                results.append(pool.apply_async(run_dill_encoded,
                                                (dill.dumps((transform, task)),)))

            again = []  # Here we temporarily keep the results that are not done yet.
            for result in results:
                # transform done?
                if result.ready():
                    # No errors?
                    if result.successful():
                        task = dill.loads(result.get())
                        logging.info("TRACER: Finished task {}".format(task))
                        # Still got stuff to do...
                        if task.wanted and task.path:
                            queue.append(task)
                        # Success, task is done but still options left, we'll skip them.
                        elif not task.wanted:
                            logging.info("TRACER: Finished task successfully: {}".format(task))
                            finished.append(task)
                        # Failure, but out of options
                        elif task.wanted and not task.path:
                            logging.info("TRACER: Could not finish task: {}".format(task))
                            finished.append(task)
                    # An error, but nothing we can do... The Transform itself
                    # should handle any case where a retry would beneficial.
                    else:
                        pass
                # Not done yet, so see again next round.
                else:
                    again.append(result)
            results = again
            # time.sleep(1)
        self.tasks = finished

    # Nice implementation with sets. However, doesn't try to minimize the set
    # of transforms in any way, so it's kinda wasteful.
    def trace(self, task):
        """Try to trace the path to solving a single Task.

        Arguments:
        task --  a Task object."""
        logging.info('TRACER: Tracing {}'.format(str(task)))
        possible = set()
        path = []
        change = True

        # Approximately works like Breadth-First-Search.
        # Try all possible transforms that add new options until we're done.
        # Check for change to prevent infinite loops
        while change:
            change = False
            # Check each possible transform.
            for req, transformlist in self.graph.items():
                # If the requirements are a subset of available and possible combined.
                print(task.available)
                if req <= (task.available | possible):
                    print('Check each transform')
                    for transform in transformlist:
                        # If the output has items not available or possible.
                        # if transform.output - (task.available | possible):
                            # We found a new option.
                        if transform not in path:
                            change = True
                            possible |= transform.output
                            path.append(transform)
                            # If wanted is a subset of possible we are done!
                            # if possible >= task.wanted:
                            #     logging.info('TRACER: Complete path found.')
                            #     return self.prune(path, task)
        logging.info('TRACER: Found path {}'.format(path))
        return self.prune(path, task)

    def prune(self, path, task):
        """Prune path to task removing unnecessary transforms.

        Arguments:
        path -- list of transforms to go from task.available to task.wanted.
        task -- Task object

        """
        newpath = []
        firstRun = True

        while newpath != path:
            if firstRun:
                firstRun = False
            else:
                path = newpath
            # Determine actual used output
            used = {req for transform in path for req in transform.required}
            used |= task.wanted
            # Only use actual transformations that are used.
            # E.g. the output and used set need to have elements in common.
            newpath = [transform for transform in path if transform.output & used]
        logging.info('TRACER: Pruned to {}'.format(path))
        return path

    def __repr__(self):
        """String representation."""
        p = '\n\t\t'.join(str(i) for i in self.transforms)
        t = '\n\t\t'.join(str(i) for i in self.tasks)
        return "Tracer:\n\tTransforms:\n\t\t{}\n\tTasks:\n\t\t{}".format(p, t)


class Transform(object):
    """Object representing a possible transformation of data."""

    def __init__(self, function, required, output, id_, weight=1):
        """Create a new Transform offering output data for required data.

        Arguments:
        function -- the function to be called with the required key arguments,
                    which returns the output as a dictionary with output keys.
        required -- Required keys (as strings) needed to run this transform.
        output   -- The output (as strings) this transform offers
        # weight   -- The priority for using this transform. If two different transforms
        #             offer the same result, the highest weight goes first. (Deprecated)
        """
        self.required = frozenset(required)
        self.output = frozenset(output)
        self.weight = weight
        self.function = function
        self.id = id_

    def __call__(self, task):
        """Obtain the output from the transform for the task.

        Output is added to the task and the wanted keys are removed.

        Arguments:
        task --  a Task object."""
        # Run transform function with required arguments
        in_ = {k: v for k, v in task.data.items() if k in self.required}
        output = self.function(**in_)
        task.history.append((in_, self.id, output))

        # Update task with new data
        for k, v in output.items():
            if 'kinetic' in k:
                task.kinetic_data.extend(v)
            elif k not in task.data:
                task.data[k] = v
            else:  # If we happen to receive any extra data, try extending it...
                if task.data[k] != v:
                    task.conflicts.append((k, v, task.data[k]))
                    # raise ValueError("Conflicting identifiers for {}:{}/{}.".format(k, v,
                    #                                                                 task.data[k]))
            # Remove data from wanted that we just included
            if k in task.wanted and 'kinetic' not in k:
                task.wanted.remove(k)
        return task

    def __repr__(self):
        """String representation."""
        return "Transform({} => {})".format(', '.join(self.required), ', '.join(self.output))


class Task(object):
    """Object representing a data task."""

    def __init__(self, data, wanted):
        """Create a new Task, with available data that wants wanted data.

        Arguments:
        data   -- Initial data dict, keys (as strings) will be added to available.
        wanted -- keys (as strings) it wants to obtain.
        """
        self.wanted = set(wanted)
        self.data = data
        self.kinetic_data = []
        self.conflicts = []
        self.path = []
        self.history = []

    def next(self):
        """Get the next transform that can be run with the available data."""
        for i, transform in enumerate(self.path):
            if transform.required <= self.available:
                break
        else:  # Loop completed, but no more viable transforms.
            raise ValueError("Leftover tasks do not have data required.")
        del self.path[i]
        return transform

    @property
    def available(self):
        """Set of avaible data keys."""
        return set(self.data.keys())

    def __repr__(self):
        """String representation."""
        return "Task({} => {})".format(', '.join(self.available), ', '.join(self.wanted))


class KineticResult(object):
    """Grouping object for the results of a kinetic parameter measurement found.

    Includes value, experimental setting and traceability information,
    missing values should be None."""

    def __init__(self,
                 kinetics_value,
                 kinetics_type,
                 kinetics_std,
                 kinetics_unit,
                 mutant,
                 organism_name,
                 kinetics_temp,
                 kinetics_ph,
                 kinetics_comments,
                 species,
                 reaction,
                 source_name,
                 source_query,
                 source_pubmed,
                 source_publication):
        """Create a kinetic result."""
        # Related to the actual value
        self.kinetics_value = kinetics_value
        self.kinetics_type = kinetics_type
        self.kinetics_std = kinetics_std
        self.kinetics_unit = kinetics_unit

        # Related to the experimental setup
        self.organism_name = organism_name
        self.mutant = mutant
        self.kinetics_temp = kinetics_temp
        self.kinetics_ph = kinetics_ph
        self.kinetics_comments = kinetics_comments

        # Related to traceability
        #   IDs in the original model
        self.species = species
        self.reaction = reaction
        #   Retrieved location.
        self.source_name = source_name
        self.source_query = source_query
        self.source_pubmed = source_pubmed
        self.source_publication = source_publication

    def __repr__(self):
        """Short representation of result."""
        return "KineticResult({} = {} for {} - {} from {})".format(self.kinetics_type,
                                                                   self.kinetics_value,
                                                                   self.reaction,
                                                                   self.species,
                                                                   self.source_name)

    @property
    def report(self):
        """Long representation of result."""
        s = """KineticResult for {}{}: {} / {} of type {}:
    Value: {} {} Std: {} {} at T = {} *C and pH = {}.
    Comments: {}
    Retrieved from {} using: "{}".""".format(self.organism_name,
                                             '' if not self.mutant else ' (Mutant)',
                                             '-' if not self.species else self.species,
                                             '-' if not self.reaction else self.reaction,
                                             self.kinetics_type,
                                             self.kinetics_value,
                                             self.kinetics_unit,
                                             self.kinetics_std,
                                             self.kinetics_unit,
                                             self.kinetics_temp,
                                             self.kinetics_ph,
                                             self.kinetics_comments,
                                             self.source_name,
                                             self.source_query)
        if (self.source_publication is not None and 'authors' in self.source_publication
                and 'year' in self.source_publication and 'title' in self.source_publication):
            s += "\n    {} - {} - {}".format(self.source_publication['authors'],
                                             self.source_publication['year'],
                                             self.source_publication['title'])
        elif self.source_pubmed:
            s += "\n    Source: Pubmed ({})".format(self.source_pubmed)

        return s


def genererateTransformFunction(in_, out, sleep=1):
    """Generate test functions which return out as a dict with in_ as arguments."""
    def f(**kwargs):
        time.sleep(sleep)
        return {i: i for i in out}
    return f


def run_dill_encoded(x):
    """Use dill to serialize the object and run it in another process.

    This is done because pickle has a lot of issues pickling methods/functions etc."""
    transform, task = dill.loads(x)
    return dill.dumps(transform(task))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    delay = 0.1
    print("Artificial delay: {}s".format(delay))
    # Some basic test cases. TODO: Add weighted transforms.
    transforms = []
    in_ = [["A", "B"], ["C", "D"], ["A", "D"], ["D"], ["E"], ["F"], ["X", "Y"], ["A", "X", "Y"]]
    out = [["X"], ["Y", "Z"], ["C"], ["E"], ["F"], ["W"], ["W"], ["W"]]
    for idx, (i, o) in enumerate(zip(in_, out)):
        f = genererateTransformFunction(i, o, delay)
        transforms.append(Transform(f, i, o, idx))
        print("({}) - {}".format(idx, transforms[-1]))

    def createtasks(n=1):
        """Create simple testing tasks."""
        tasks = []
        for i in range(n):
            # Straightforward task
            t1 = Task({i: i for i in ["A", "B", "C", "D"]}, ["X", "Y", "Z"])
            # (0) A + B => X
            # (1) C + D => Y + Z

            # 1 Intermediate step
            t2 = Task({i: i for i in ["A", "B", "D"]}, ["X", "Y", "Z"])
            # (0) A + B => X
            # (2) A + D => C
            # (1) C + D => Y + Z

            # 2 Steps
            t3 = Task({i: i for i in ["A", "B", "D"]}, ["W"])
            # (0) A + B => X
            # (2) A + D => C
            # (1) C + D => Y + Z
            # (3) X + Y => W
            tasks += [t1, t2, t3]
        return tasks

    n = 1

    # Max cores
    m1 = Tracer(createtasks(n), transforms)
    t0 = time.clock()
    print("Async 4 processes.")
    print("Input: {} tasks".format(n * 3))
    m1.run_async(processes=4, ignore_error=True)
    s = len([t for t in m1.tasks if not t.wanted])
    print("Results: {}/{} successful".format(s, n * 3))
    print(time.clock() - t0)

    # # Single core
    # m2 = Tracer(createtasks(n), transforms)
    # t0 = time.clock()
    # print("Async 1 process.")
    # print("Input: {} tasks".format(n * 3))
    # m2.run_async(processes=1, ignore_error=True)
    # s = len([t for t in m2.tasks if not t.wanted])
    # print("Results: {}/{} successful".format(s, n * 3))
    # print(time.clock() - t0)

    # # Non parallel
    # m3 = Tracer(createtasks(n), transforms)
    # t0 = time.time()
    # print("Sync.")
    # print("Input: {} tasks".format(n * 3))
    # m3.run()
    # s = len([t for t in m3.tasks if not t.wanted])
    # print("Results: {}/{} successful".format(s, n * 3))
    # print(time.time() - t0)
