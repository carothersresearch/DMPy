Author: Rik van Rosmalen
Contact: rik.vanrosmalen@wur.nl / rikpetervanrosmalen@gmail.com

Contents:
    - File overview
    - Pipeline
        * Installation
        * Detailed Overview
        * Usage

------------------------------------------------------
                    File overview
------------------------------------------------------
Important files:
    enviroment.yml - Conda enviroment file.

    Pipeline files: (See details per file in next section)
        pipeline.py
        sbmlwrap.py
        databases.py
        datatracer.py
        identifiers.py

    shelve_cache/ 
        Portable caching databases with parameter values already retrieved online.
        Online requests to Sabio-Rk / Brenda will be cached here.
        Values for other databases (eQuilibrator / Rhea / MetaNetX / MetaCyc) will have to be 
        generated manually first. 
        To use these data sources, see NOTE_DATA.txt
        For more detail see the example scripts for MetaNetX and MetaCyc, and the
        database script for eQuilibrator and Rhea.

    Example files:
        example.py - Shows how to generate and use values to simulate ensembles.
        plots.py - Shows how to analyise the data produced from `example.py`

    Other files:
        optimized_functions.pyx - A single Cython function to speed up balancing of large networks
        random_model.py - Example script on generating small random networks
        load_metacycdb.py - Example script on loading metacyc data.
        load_metanetx.py - Example script on loading metanetx data.

    Models/*.xml:
        Some example models are included. For the latest version of these models,
        visit the biomodels repository website: https://www.ebi.ac.uk/biomodels-main

------------------------------------------------------
                      Pipeline
------------------------------------------------------
Installation
---
The easiest way to get all python modules up and running is using the anaconda 
python distribution (get version 2.x) which comes with precompiled packages 
for windows/OS x or Linux. You can find it at: https://www.continuum.io/

Otherwise, all modules should be installable with pip, the python package 
manager. 1) Install python from python.org (Version 2)
2) You can find pip at: https://pip.pypa.io/en/stable/installing/

Using either conda or pip install the following modules:
    Install globally: `sudo pip install $module$`
    Install locally: `pip install $module$ -user` (Does not require admin)
    Get:
        numpy 
        scipy
        sympy
        soappy (For accessing Brenda) -
            (the requirement wstools has some small bugs in the current published version on pip 
             as of 1/3/2017, but the latest github version worked.)
        requests (For accessing Sabio-RK)
        libsbml (For reading/writing sbml, the package name is python-libsbml)
        libroadrunner (For fast numerical ODE simulation)
        Pint (For easy unit conversion)
        networkx (Random model generation)

    Note: libroadrunner is not strictly required, since direct generation of the ODE system which
          can be integrated using scipy.ode.integrate using sympy is supported. However, this is 
          notably slower, so the examples use the more mature libroadrunner library.

    Optional:
        cython (For speeding up part of the balancing, requires a C compiler)

    In addition:
        run `python setup_cython.py build_ext --inplace`

---
Detailed Overview
---
Pipeline (`pipeline.py`):
    The pipeline consists of three main parts which can be used separately 
    or chained together:
        - Gather parameters
        - Balance parameters
        - Construct model
        - Simulate (See examples)

    `pipeline.py` can be used as a standalone script through:
        `python pipeline.py $path to sbml$ $Species name$ $Output directory$`
    Example:
        `python pipeline.py data/iTO977_v1.00_raven.xml "Saccharomyces cerevisiae" output-s.cerivisea/`
    Note: `pipeline.py` can use both Lubitz method or my reimplementation. Use Lubitz' 
          method by specifying `-lubitz`. 
          The source code for Lubitz' method can be found on parameterbalancing.net
    For more options, see the documentation in the file or call `python pipeline.py -h`

Parameter gathering (`datatracer.py`, `databases.py`, `identifiers.py`):
    These three files are the core of the data tracing.
    +++
    In `datatracer.py` we can find the main classes:
    Tracer:
        The object that runs the gathering. Receives Tasks and Transforms,
        adds paths to each task and applies transforms to get the data for tasks.
        Note: Multiprocessed running was implemented but is not functional for the pipeline, 
        due to the limitations of the caching databases, which are not compatible with
        simultaneous write access!
    Task:
        An object representing a task, has information and wanted information.
        Keeps track of its own data paths obtained from the tracer and whether it can
        run the path. (i.e. does it have the required data for the next transform?)
        Tasks are made in the pipeline after parsing the SBML file.
        Also keeps track of it's history for debugging purposes.
    Transform:
        An object representing a data transformation (function), takes a set of inputs 
        and gives a possible output (not guaranteed!)
        Can be applied to a task, where it will call the supplied function using the 
        data of task, and add the output to data.
    +++
    In `databases.py` the transforms are made. Each class represents a database,
    but they all have a function: getTransforms, which gets the list of transforms
    which this database is able to make.
    +++
    In `identifiers.py` the standardized names for the identifiers can be found back. 
    Use these for consistency instead of defining strings everywhere in your code.
    +++
    Note: Brenda requires an account for API access.
    +++
    Note: Brenda and SabioRK are rate-limited by default, to prevent spamming the API.
    +++
    Note: Most databases classes are cached in an of-line shelve database to prevent 
          repeated slow API access. However, this database does NOT support simultaneous write access! 
          If multiple scripts are writing to the same database, it will get corrupted! 
          Shelve databases are portable, but see the python documentation for details. 
          (Notably, OS X can have some trouble - see:
           http://stackoverflow.com/questions/814041/how-to-fix-the-broken-bsddb-install-in-the-default-python-package-on-mac-os-x-10)
    +++
    Note: The metacyc shelve db is generated from their file dump, but can be recreated
          using `load_metacycdb.py`. The same holds for metanetx and `load_metanetx.py`.

   
Model reading/construction (`sbmlwrap.py`)
    `sbmlwrap.py` is a convenience wrapper around the powerfull but hard to use SBMLlib library.
    It can parse the SBML files and extract reactions, compounds and identifiers in 
    a more convenient format.
    Some of the more useful functions:
        - Loading the model:
            - Model($path to SBML file$)
            - Model.from_string($SBML file as a string$)
        - Convert to model with kinetics:
            - Model.kineticize (To add kinetics)
            - Model.get_formulas (To get Sympy formulas for the model)
            - Model.get_ode_function (To get a python usable ODE function)
        - Writing the model
            - Model.writeSBML (Output the model as SBML)
    +++
    Note: If the model does not parse your identifiers correctly, have a look at 
          the various parse functions inside this file.

Balancing (`balancer.py`)
    `balancer.py` implements the balancing of Lubitz et al. (2010), See their
    appendix and paper for a more detailed explanation.
    The main code is a straightforward translation of the mathematics in the 
    appendix, and is documented with several test cases inside the file.
