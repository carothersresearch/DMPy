#!/usr/bin/env python
"""
Scripts to reduce, parametrize and balance a biological model.

Every database should expose it's possible data transformation functions trough a function
.getTransforms()
This should return a tuple of one or more Transform objects.


Author: Rik van Rosmalen
"""
from __future__ import division
from __future__ import print_function

import functools
import collections
import shelve
import logging
import hashlib
import re
import time
import math
import os.path
import itertools

from SOAPpy import SOAPProxy
import requests

from datatracer import Transform, KineticResult
from sbmlwrap import Model
from identifiers import Identifiers

# Base location of shelve caches.
CACHE_DIR = os.path.join(os.path.curdir, 'shelve_cache')


# -----------------------------------------------------------------------------
#                              Custom exceptions
# -----------------------------------------------------------------------------
class DataBaseError(Exception):
    """Base exception for errors with the databases."""

    pass


class BrendaError(DataBaseError):
    """Exception for errors from the Brenda soap API."""

    pass


class RheaError(DataBaseError):
    """Exception for errors from the rhea data."""

    pass


class EquilibratorError(DataBaseError):
    """Exception for errors from Equilibrator data."""

    pass


class MetacycError(DataBaseError):
    """Exception for errors from Metacyc data."""

    pass


class MetaNetXError(DataBaseError):
    """Exception for errors from MetaNetX data."""

    pass


# -----------------------------------------------------------------------------
#                                 Decorators
# -----------------------------------------------------------------------------

def rate_limited(seconds):
    """Decorator to make subsequent calls wait at least x seconds.

    Can be used to rate limit outside API calls.

    If used in conjunction with `cached`, make sure to apply rate_limited first,
    so it doesn't rate limit the local cached calls!
    e.g: cached(location)(rate_limited(5)(request))

    Arguments:
    seconds -- time to wait between requests in seconds.
    """
    def loc_wrapper(f):
        logging.info("RATE: {} limited to {} calls/sec.".format(f.__name__,
                                                                seconds))
        last_time = [0.0]

        @functools.wraps(f)
        def rate_wrapper(*args):
            elapsed = time.clock() - last_time[0]
            if seconds > elapsed:
                time.sleep(seconds - elapsed)
            results = f(*args)
            last_time[0] = time.clock()
            return results
        return rate_wrapper
    return loc_wrapper


def cached(cache_location):
    """Decorator to generate a persistent shelve cache for a function.

    Can be used to prevent repeated (slow) outside API calls.

    If used in conjunction with `cached`, make sure to apply rate_limited first,
    so it doesn't rate limit the local cached calls!

    Arguments:
    cache_location -- path to file location, the actual file will have '.db' as extension.
    """
    def loc_wrapper(f):
        logging.info("CACHE: {} cache at:{}.".format(f.__name__, cache_location))

        @functools.wraps(f)
        def cache_wrapper(*args):
            # protocol -1 should use the most up to date pickle protocol.
            cache = shelve.open(cache_location, protocol=-1)
            # Convert arguments to a key
            args_string = str(args)
            # print cache_location, args_string # If you get UnpicklingError, check here
            # See which entry is problematic and try deleting if possible.
            try:
                result = cache[args_string]
                logging.info("CACHE: retrieved from cache ({}): {}.".format(cache_location,
                                                                            args_string))
            except KeyError:
                result = f(*args)
                # print cache_location, args_string
                # print result
                cache[args_string] = result
                logging.info("CACHE: saved in cache ({}): {}.".format(cache_location,
                                                                      args_string))
            cache.close()
            return result
        return cache_wrapper
    return loc_wrapper


# -----------------------------------------------------------------------------
#                              Helper functions
# -----------------------------------------------------------------------------

def try_shelve(shelve, key):
    """Try to access a shelve key. If not there, return None."""
    try:
        return shelve[key]
    except KeyError:
        return None


def match_compounds(*names):
    """Compare an arbitrary amount of compound names, returning True if all compounds match.

    Requires metacyc_names_compounds.shelve file."""
    # TODO: Clean up this function.
    db = shelve.open(os.path.join(CACHE_DIR, "metacyc_names_compounds.shelve"), protocol=-1)
    names = [name.lower() for name in names]

    for i, name in enumerate(names):
        if isinstance(name, unicode):
            names[i] = name.encode('ascii', 'ignore')

    orig_name = names[0].lower()
    orig_id = try_shelve(db, orig_name)
    result = True
    for name in names[1:]:
        # Try match of lowercase strings first.
        if name.lower() != orig_name:
            # Also try if its in metacyc synonyms.
            next_id = try_shelve(db, name)
            if orig_id != next_id or next_id is None:
                result = False
                break
    db.close()
    return result


# -----------------------------------------------------------------------------
#                                 Classes
# -----------------------------------------------------------------------------
class Brenda(object):
    """Object to access the Brenda database.

    See the soap API here: http://www.brenda-enzymes.org/soap.php
    Functions can be called using 'request', which will add your user details,
    compose and send the soap query.
    Output will be parsed into a nested list of entries, consisting of key and
    value pairs in a dictionary.
    """

    def __init__(self, user, password,
                 shelve_cache=os.path.join(CACHE_DIR, "brenda.shelve"),
                 delay=5, wsdl="http://www.brenda-enzymes.org/soap/brenda.wsdl"):
        """Initialize the soap server and set up the user.

        Arguments:
        user -- e-mail address of the user (you have to sign up at the brenda website.)
        password -- sha256 hexdigest of your password (hashlib.sha256(password).hexdigest())
        shelve_cache -- cache location to save Brenda requests offline.
                        Set to False to not use a cache.
        delay -- seconds to wait in-between API calls. set to False to not use a delay.
        wsdl -- wsdl address of Brenda database. [To explore functions exposed check
                                                  the WSDL object under self.client]
        """
        self.wsdladdress = wsdl
        # TEMPORARY FIX
        # Since the WSDL points to the wrong address, manually set the endpoint.
        # self.client = WSDL.Proxy(self.wsdladdress)
        self.client = SOAPProxy("https://www.brenda-enzymes.org/soap/brenda_server.php")

        self.user = user
        self.password = hashlib.sha256(password).hexdigest()
        self.shelve_cache = shelve_cache

        logging.info("BRENDA: Set up Brenda database for user: "
                     "{} using {}.".format(self.user, self.wsdladdress))

        # Rate-limit the request so it doesn't spam the outside server.
        if delay > 0:
            self.request = rate_limited(delay)(self.request)
        # Wrap the rate-limited request in shelve cache if a location is provided.
        if self.shelve_cache:
            self.request = cached(shelve_cache)(self.request)

    def compose_parameter_string(self, parameters):
        """Compose the parameter string for the soap query, adding in the user and password.

        Arguments:
        parameters -- list of (key, value) parameters
        """
        parameters = '#'.join('*'.join(item) for item in parameters)
        return ','.join([self.user, self.password, parameters])

    def parse_output(self, output):
        """Parse the output from a Brenda soap API request call.

        Arguments:
        output -- Brenda soap api request output. Will be split into
                  a list of dictionary entries.
        """
        parsed = []
        entries = output.split('!')  # ! is used to delimit entries
        for entry in entries:
            fields = entry.split('#')  # # delimits the fields
            d = {}

            for field in fields:
                if not field.strip():  # Skip empty fields
                    continue
                # * delimits the values.
                # However, '*' is an allowed character in the value,
                # so we only split on the first occurrence here.
                # print output
                k, v = (i.strip() for i in field.split('*', 1))
                if k and v:  # Skip empty values.
                    if k in d:
                        logging.warning("Duplicate key in output from Brenda request {} with "
                                        "values {} and {}".format(k, d[k], v))
                    d[k] = v

            parsed.append(d)
        return parsed

    def request(self, func, parameters):
        """Query the soap server with func called with parameters.

        The current user and password are included automatically.

        Arguments:
        func -- String of the soap server method to be called
        parameters -- list of (key, value) parameters
        """
        # Temporary fix since WSDL is not up to date.
        # try:
        #     f = getattr(self.client, func)
        # except AttributeError:
        #     raise BrendaError("Brenda does not support request of type {}".format(func))

        pars = self.compose_parameter_string(parameters)
        # out = f(pars)

        if func == "getKmValue":
            out = self.client.getKmValue(pars)
        elif func == "getKiValue":
            out = self.client.getKiValue(pars)
        elif func == "getTurnoverNumber":
            out = self.client.getTurnoverNumber(pars)
        # Note: It seems that Brenda actually only responds to the first number of the request.
        # Leading to returning the same references over and over again (i.e. reference ID: 1-9)
        # This has to be fixed on Brenda's backend. For now, just manually check the webpage
        # which seems to work fine regardless.
        elif func == "getReferenceById":
            out = self.client.getReferenceById(pars)

        if out == 'Activation required, please check your emails!':
            raise BrendaError("User not activated: {}".format(self.user))
        if out == 'Password incorrect! Please try again.':
            raise BrendaError("Incorrect password for user {}".format(self.user))
        if out == 'Unknown user. Please register. www.brenda-enzymes.org/register.php':
            raise BrendaError("Unknown user: {}".format(self.user))
        elif out == '':
            # No output, can be either a malformed query or just no results.
            # However, we will leave this up to the caller to figure out as the first is
            # an error and the second is case correct but we cannot differentiate.
            return {}

        return self.parse_output(out)

    def findTemp(self, s):
        """Helper function to retrieve the temperature from the comment section."""
        m = re.search(r"([0-9]+\.{0,1}[0-9]*)\s{0,4}&deg;\s{0,4}C", s, re.IGNORECASE)
        if m:
            return m.group(1)
        else:
            return None

    def findpH(self, s):
        """Helper function to retrieve the pH from the comment section."""
        m = re.search(r"pH\s{0,4}([0-9]+\.[0-9]+)", s)
        if m:
            return m.group(1)
        else:
            return None

    def findMutant(self, s):
        """Helper function to check if mutant from the comment section."""
        m = re.search(r"mutant", s, re.IGNORECASE)
        return bool(m)

    def getTransforms(self):
        """Create the data transforms that Brenda can provide."""
        def template(k_val, k_get, k_species, k_kin):

            def f(**kwargs):
                try:
                    out = self.request(k_get, [("ecNumber", kwargs[Identifiers.enzyme_EC]),
                                               ("organism", kwargs[Identifiers.organism_name])])
                except BrendaError:
                    raise
                except Exception:
                    # print last
                    raise
                    # logging.error("BRENDA: Request failed: {}".format(e))
                    # return {}

                results = []
                for l in out:
                    # Return only if the species match.
                    if match_compounds(l[k_species], kwargs['species']):
                        # Brenda has some -999 lines sometimes for some reason.
                        if float(l[k_val]) < 0:
                            logging.info("BRENDA: Invalid value encountered {} < 0!".format(k_val))
                            continue
                        try:
                            c = l['commentary']
                        except KeyError:
                            c = ''

                        # Construct the readable query used.
                        query = "Organism({}), EC({})".format(kwargs[Identifiers.organism_name],
                                                              kwargs[Identifiers.enzyme_EC])

                        # Standard units of Brenda / Reaction only for things that have a reaction.
                        if k_kin == Identifiers.kinetics_km or k_kin == Identifiers.kinetics_ki:
                            unit = 'mM'
                        elif k_kin == Identifiers.kinetics_kcat:
                            unit = '1/s'
                        else:
                            unit = None

                        # Try to request source information.
                        brenda_publication_id = l['literature']
                        publication = self.request('getReferenceById', brenda_publication_id)[0]
                        if 'pubmedId' in publication:
                            pubmed = publication['pubmedId']
                        else:
                            pubmed = None

                        # Fill out the results
                        result = KineticResult(kinetics_value=l[k_val],
                                               kinetics_type=k_kin,
                                               kinetics_std=None,
                                               kinetics_unit=unit,
                                               organism_name=kwargs[Identifiers.organism_name],
                                               mutant=self.findMutant(c),
                                               kinetics_temp=self.findTemp(c),
                                               kinetics_ph=self.findpH(c),
                                               kinetics_comments=c,
                                               species=kwargs[Identifiers.species],
                                               reaction=kwargs[Identifiers.reaction],
                                               source_name='Brenda',
                                               source_query=query,
                                               source_pubmed=pubmed,
                                               source_publication=publication)
                        results.append(result)
                if not results:
                    logging.info("BRENDA: Request failed: No species match in output.")
                return {k_kin: results}
            return f

        f1 = template('kmValue', 'getKmValue', 'substrate', Identifiers.kinetics_km)
        p1 = Transform(f1, [Identifiers.enzyme_EC, Identifiers.organism_name, Identifiers.species,
                       Identifiers.reaction], [Identifiers.kinetics_km], "Brenda (km)")

        f2 = template('kiValue', 'getKiValue', 'inhibitor', Identifiers.kinetics_ki)
        p2 = Transform(f2, [Identifiers.enzyme_EC, Identifiers.organism_name, Identifiers.species,
                       Identifiers.reaction], [Identifiers.kinetics_ki], "Brenda (ki)")

        f3 = template('turnoverNumber', 'getTurnoverNumber', 'substrate', Identifiers.kinetics_kcat)
        p3 = Transform(f3, [Identifiers.enzyme_EC, Identifiers.organism_name, Identifiers.species,
                       Identifiers.reaction], [Identifiers.kinetics_kcat], "Brenda (kcat)")

        return [p1, p2, p3]


class Equilibrator(object):
    """Object to access eQuilibrator data."""

    def __init__(self, data_location=None,
                 shelve_cache=os.path.join(CACHE_DIR, "equilibrator.shelve")):
        """Initialize the Equilibrator data.

        Arguments:
        data_location -- file path to data location
        shelve_cache -- cache location to save Equilibrator requests.
                        Set to False to not use a cache.
        """
        self.data_location = data_location
        self.shelve_cache = shelve_cache
        self.R = 8.3144598
        # Retrieve the conditions. This should be the same for the whole file.
        if self.data_location is None:
            self.T = 298.15
            self.pH = 7.0
        else:
            with open(self.data_location, 'r') as infile:
                infile.readline()  # Could possible extract units here.
                # However, they are standard units already. (G in kJ/mol, T in K, I in mM)
                _, _, _, pH, I, T, _ = infile.readline().split(',')
                self.T = T
                self.pH = pH

        if self.shelve_cache:
            self.retrieve = cached(shelve_cache)(self.retrieve)

    def retrieve(self, keggID):
        """Retrieve the dG0 and standard deviation from the data file.

        Arguments:
        keggID -- keggID of the reaction
        """
        with open(self.data_location, 'r') as infile:
            infile.readline()  # Skip the header
            for line in infile:
                if line.startswith(keggID):
                    _, dG0, sG0, _, _, _, Note = line.split(',')
                    if Note.strip() == "uncertainty is too high":
                        raise EquilibratorError("No value found: Uncertainty too high")
                    else:
                        return float(dG0), float(sG0)
            raise EquilibratorError("No value found: KeggID not found")

    def calculateKeq(self, dG0):
        """Calculate the equality constant from dG0.

        Arguments:
        dG0 -- change in Gibbs free energy (kJ/mol) of the reaction.
        """
        try:
            if dG0 > 500:
                dG0 = 500
                logging.info('EQUILIBRATOR: DG0 Value to high/low, set to 500')
            elif dG0 < -500:
                dG0 = -500
                logging.info('EQUILIBRATOR: DG0 Value to high/low, set to -500')
            k = math.pow(math.e, (float(dG0) * 1000 / (-float(self.R) * float(self.T))))
        except OverflowError:
            raise EquilibratorError("An overflow occured in calculating the Keq.")
            # Some have crazy values... E.G. R05617 with -2767.0 kJ
        return k

    def getTransforms(self):
        """Create the data transforms that eQuilibrator can provide."""
        def f(**kwargs):
            try:
                dG0, sG0 = self.retrieve(kwargs[Identifiers.kegg_reaction])
                Keq = self.calculateKeq(dG0)
            except EquilibratorError:
                return {}
            stdev = None
            query = "KeggID ({})".format(kwargs[Identifiers.kegg_reaction])

            # Fill out the results
            result = KineticResult(kinetics_value=Keq,
                                   kinetics_type=Identifiers.kinetics_keq,
                                   kinetics_std=stdev,
                                   kinetics_unit='-',
                                   organism_name=None,
                                   mutant=None,
                                   kinetics_temp=self.T,
                                   kinetics_ph=self.pH,
                                   kinetics_comments=None,
                                   species=None,
                                   reaction=kwargs[Identifiers.reaction],
                                   source_name='eQuilibrator',
                                   source_query=query,
                                   source_pubmed=None,
                                   source_publication=None)

            return {Identifiers.kinetics_keq: [result]}

        p = Transform(f, [Identifiers.kegg_reaction, Identifiers.reaction],
                      [Identifiers.kinetics_keq], "eQuilibrator (keq)")
        return [p]


class Rhea(object):
    """Object to access rhea cross reference data."""

    def __init__(self, data_location=None,
                 shelve_db=os.path.join(CACHE_DIR, "rhea.shelve")):
        """Initialize the Rhea data.

        Arguments:
        data_location -- file path to data location folder
        shelve_db     -- db location to save parsed data.

        """
        self.shelve_db = shelve_db
        if not os.path.exists(self.shelve_db):
            if data_location is not None:
                self.parse(data_location)
            else:
                raise RheaError("Please provided either a data file"
                                " or a cached db of the rhea xrefs.")

    def parse(self, filename):
        """Parse the rhea2xrefs file."""
        keys = {'METACYC': Identifiers.metacyc_reaction_id, 'UNIPROT': Identifiers.uniprot_id,
                'KEGG_REACTION': Identifiers.kegg_reaction, 'EC': Identifiers.enzyme_EC,
                'UNIPATHWAY': Identifiers.unipathway_reaction_id, 'MACIE': Identifiers.macie_id,
                'ECOCYC': Identifiers.ecocyc_reaction_id,
                'REACTOME': Identifiers.reactome_reaction_id}

        def blocks(iterator):
            """Take an iterator and return blocks.

            All entries in a block have the same master_id. Filters empty lines.
            """
            temp = []
            old_master_id = ''
            for line in iterator:
                if line.strip():
                    master_id = line.split()[2]
                    if master_id != old_master_id:
                        old_master_id = master_id
                        yield temp
                        temp = [line]
                    else:
                        temp.append(line)

        data = collections.defaultdict(dict)
        with open(filename) as infile:
            infile.readline()  # Skip header
            for block in blocks(infile):
                for line in block:  # Not really needed any more.
                    rhea_id, direction, master_id, id_, db = line.split()
                    # if db == 'UNIPROT' or db == 'UNIPATHWAY':
                    #     continue
                    # From other id to Rhea
                    data[keys[db]][id_] = master_id
                    # From Rhea to other id
                    data[master_id][keys[db]] = id_
                #     # Add rhea master id to other database references.
                #     if ('rhea_sub_id', rhea_id) not in l:
                #         l.append(('rhea_sub_id', rhea_id))
                #     l.append((db, id_))
                #     # Add other database to rhea master id references.
                #     data[(db, id_)] = ('rhea_id', master_id)
                # if block:
                #     data[('rhea_id', master_id)] = l

        db = shelve.open(self.shelve_db, protocol=-1)
        for key, value in data.items():
            db[key] = value
        db.close()

    def retrieve(self, database, id_):
        """Retrieve the cross reference from the shelve database."""
        db = shelve.open(self.shelve_db, protocol=-1)
        if database == 'rhea_id':
            value = try_shelve(db, id_)
        else:
            data = try_shelve(db, database)
            try:
                if id_ in data:
                    value = data[id_]
                else:
                    value = None
            except TypeError:
                print(db)
                print(database)
                print(data)
                db.close()
                raise
        db.close()
        return value

    def getTransforms(self):
        """Return all transforms using rhea cross reference data."""
        out = {'METACYC': Identifiers.metacyc_reaction_id, 'UNIPROT': Identifiers.uniprot_id,
               'KEGG_REACTION': Identifiers.kegg_reaction, 'EC': Identifiers.enzyme_EC,
               'MACIE': Identifiers.macie_id, 'UNIPATHWAY': Identifiers.unipathway_reaction_id,
               'ECOCYC': Identifiers.ecocyc_reaction_id,
               'REACTOME': Identifiers.reactome_reaction_id}
        p = []

        # Rhea id -> Other ids
        def f(**kwargs):
            raw = self.retrieve('rhea_id', kwargs[Identifiers.rhea_id])
            d = {}
            if raw is not None:
                for db, id_ in raw.items():
                    if db in out.values():
                        d[db] = id_
            return d

        skip = set(['UNIPROT'])
        p.append(Transform(f, [Identifiers.rhea_id], set(out.values()) - skip, 'Rhea (X-id)'))

        # Other id -> Rhea id
        def template(db):
            def f(**kwargs):
                d = {}
                raw = self.retrieve(db, kwargs[db])
                if raw:
                    d['rhea_id'] = raw
                return d
            return f

        # All but enzyme_EC can also go in different direction.
        skip = set(['EC'])
        p += [Transform(template(db_in), [db_in], [Identifiers.rhea_id], 'Rhea (X-id)')
              for db_in in set(out.values()) - skip]

        return p


class Metacyc(object):
    """Object to access Metacyc flat file data."""

    def __init__(self, shelve_path_prefix=os.path.join(CACHE_DIR, 'metacyc'),
                 data_file_path=None):
        """Initialize the Metacyc data.

        Arguments:
        shelve_path_prefix -- prefix for shelve files
        data_file_path -- file path for the Metacyc flat files to generate shelves.
                          (only required if shelves haven't been generated yet.)
        """
        # Check for availability of either data or a previous database.
        self.shelve_data = shelve_path_prefix + '_data.shelve'
        self.shelve_compound_names = shelve_path_prefix + '_names_compounds.shelve'
        self.shelve_reaction_names = shelve_path_prefix + '_names_reaction.shelve'
        self.shelve_enzyme_reaction_names = (shelve_path_prefix +
                                             '_names_enzyme_reaction.shelve')
        self.shelve_identifiers = shelve_path_prefix + '_identifiers.shelve'

        # TODO: Fix different file extensions.

        if not self.checkfiles():
            if data_file_path is None:
                raise MetacycError("Provide either a path to the flat file data "
                                   "or to a previously generated shelve database.")
            else:
                self.generate_shelve_db(shelve_path_prefix, data_file_path)

    def checkfiles(self):
        """Check if all the shelve databases with Metacyc output are available."""
        files = (self.shelve_data, self.shelve_compound_names, self.shelve_reaction_names,
                 self.shelve_enzyme_reaction_names, self.shelve_identifiers)
        shelve = True
        has_db_extension = False
        for path in files:
            if not os.path.isfile(path):
                if os.path.isfile(path + '.db'):
                    has_db_extension = True
                else:
                    shelve = False
        if has_db_extension:
            self.shelve_data += '.db'
            self.shelve_compound_names += '.db'
            self.shelve_reaction_names += '.db'
            self.shelve_enzyme_reaction_names += '.db'
            self.shelve_identifiers += '.db'
        return shelve

    def generate_shelve_db(self, shelve_path_prefix, data_file_path):
        """Create the shelve databases from Metacyc flat files."""
        import load_metacycdb
        files = 'compounds.dat', 'enzrxns.dat', 'reactions.dat'
        c, e, r = (os.path.join(data_file_path, f) for f in files)
        load_metacycdb.main(c, e, r, shelve_path_prefix)

    def retrieve_id_by_name(self, name, type_):
        """Retrieve the Metacyc id of a compound, reaction or enzyme_reaction."""
        name = name.lower()
        if type_ == 'compound':
            db = self.shelve_compound_names
        elif type_ == 'reaction':
            db = self.shelve_reaction_names
        elif type_ == 'enzyme_reaction':
            db = self.shelve_enzyme_reaction_names
        else:
            raise MetacycError("Invalid data type for retrieval by name: {}".format(type_))

        db = shelve.open(db, protocol=-1)
        value = try_shelve(db, name)
        db.close()
        return value

    def retrieve_id_by_id(self, id_, db):
        """Retrieve the Metacyc id corresponding to the id from another database."""
        other = db
        db = shelve.open(self.shelve_identifiers, protocol=-1)
        value = try_shelve(db, other)
        if value is not None and id_ in value:
            value = value[id_]
        else:
            value = None
        db.close()
        return value

    def retrieve_data(self, id_):
        """Retrieve the Metacyc data by the Metacyc id."""
        db = shelve.open(self.shelve_data, protocol=-1)
        value = try_shelve(db, id_)
        db.close()
        return value

    def getTransforms(self):
        """Return all transforms using Metacyc data."""
        # TODO: Might need some refactoring...
        f = None
        # List of all transforms
        p = []

        # These are all the identifiers extracted from the data, minus some useless ones.
        compound_dbs = [(Identifiers.drugbank_id, 402), (Identifiers.metabolights_id, 1792),
                        (Identifiers.chemspider_id, 4189), (Identifiers.cas_id, 1492),
                        (Identifiers.bigg_metabolite_id, 801), (Identifiers.lipid_maps_id, 396),
                        (Identifiers.pubchem_id, 11328), (Identifiers.hmdb_id, 2731),
                        (Identifiers.kegg_compound, 4975), (Identifiers.chebi_id, 6760),
                        (Identifiers.nci_id, 218), (Identifiers.smiles, 11361),
                        (Identifiers.inchi_key, 11850), (Identifiers.inchi, 11849)]
        reaction_dbs = [(Identifiers.kegg_reaction, 4526), (Identifiers.uniprot_id, 24900),
                        (Identifiers.rhea_id, 4233), (Identifiers.pir_id, 447)]

        # Base transform from names -> metacyc id (Compound)
        def f(**kwargs):
            d = {}
            id_ = self.retrieve_id_by_name(kwargs[Identifiers.species].lower(), "compound")
            if id_ is not None:
                d[Identifiers.metacyc_compound_id] = id_
            return d

        p.append(Transform(f, [Identifiers.species], [Identifiers.metacyc_compound_id],
                 "Metacyc (X-id)"))

        # Base transform from names -> metacyc id (Reaction)
        def f(**kwargs):
            d = {}
            # See if we can the name in reaction or enzyme reactions.
            id1 = self.retrieve_id_by_name(kwargs[Identifiers.reaction], "reaction")
            if id1 is not None:
                d[Identifiers.metacyc_reaction_id] = id1
            id2 = self.retrieve_id_by_name(kwargs[Identifiers.reaction], "enzyme_reaction")
            if id2 is not None:
                d[Identifiers.metacyc_enzyme_reaction_id] = id2

            # See if we can find a crosslink instead...
            if id1 is not None and id2 is None:
                db = 'enzyme_reaction'
                id2 = self.retrieve_id_by_id(id1, db)
                if id2 is not None:
                    d[Identifiers.metacyc_enzyme_reaction_id] = id2
            elif id1 is None and id2 is not None:
                db = 'reaction'
                id1 = self.retrieve_id_by_id(id2, db)
                if id1 is not None:
                    d[Identifiers.metacyc_reaction_id] = id1
            return d

        p.append(Transform(f, [Identifiers.reaction], [Identifiers.metacyc_reaction_id,
                                                       Identifiers.metacyc_enzyme_reaction_id],
                           "Metacyc (X-id)"))

        name_mapping = {Identifiers.chemspider_id: 'chemspider',
                        Identifiers.drugbank_id: 'drugbank',
                        Identifiers.lipid_maps_id: 'lipid_maps',
                        Identifiers.nci_id: 'nci',
                        Identifiers.pubchem_id: 'pubchem',
                        Identifiers.bigg_metabolite_id: 'bigg',
                        Identifiers.cas_id: 'cas',
                        Identifiers.chebi_id: 'chebi',
                        Identifiers.hmdb_id: 'hmdb',
                        Identifiers.inchi_key: 'inchi-key',
                        Identifiers.inchi: 'inchi',
                        Identifiers.kegg_compound: 'ligand-cpd',
                        Identifiers.metabolights_id: 'metabolights',
                        Identifiers.smiles: 'smiles',
                        # reaction dbs
                        Identifiers.kegg_reaction: 'ligand-rxn',
                        Identifiers.rhea_id: 'rhea',
                        Identifiers.pir_id: 'pir',
                        Identifiers.uniprot_id: 'uniprot',
                        # Self
                        Identifiers.metacyc_enzyme_reaction_id: 'enzymatic-reaction',
                        Identifiers.metacyc_reaction_id: 'reaction'
                        }

        # other id -> metacyc id
        def template(id_in, id_out):
            def f(**kwargs):
                d = {}
                id_ = self.retrieve_id_by_id(kwargs[id_in], name_mapping[id_in])
                if id_ is not None:
                    d[id_out] = id_
                return d
            return f

        p += [Transform(template(db_id, Identifiers.metacyc_compound_id),
                        [db_id], [Identifiers.metacyc_compound_id], "Metacyc (X-id)")
              for db_id, _ in compound_dbs]
        p += [Transform(template(db_id, Identifiers.metacyc_reaction_id),
                        [db_id], [Identifiers.metacyc_reaction_id], "Metacyc (X-id)")
              for db_id, _ in reaction_dbs]

        # metacyc id -> metacyc data / other id
        def template(id_, db_out, data_out):
            def f(**kwargs):
                d = {}
                raw = self.retrieve_data(kwargs[id_])
                if raw is None:
                    return d
                else:
                    for db_id in db_out:
                        db = name_mapping[db_id]
                        if db in raw:
                            d[db_id] = raw[db]
                    for key, value in data_out.items():
                        if key in raw:
                            d[value] = raw[key]
                return d
            return f

        # compound metacyc id -> other id + other info (Compound)
        compound_dbs_out = [db for db, count in compound_dbs]
        compound_out = {}
        f = template(Identifiers.metacyc_compound_id, compound_dbs_out, compound_out)
        p.append(Transform(f, [Identifiers.metacyc_compound_id],
                           compound_dbs_out + list(compound_out.values()),
                           "Metacyc (X-id)"))

        # reaction metacyc id -> other id + other info (Reaction)
        reaction_dbs_out = [db for db, count in reaction_dbs]
        reaction_out = {'ec-number': Identifiers.enzyme_EC}
        f = template(Identifiers.metacyc_reaction_id, reaction_dbs_out, reaction_out)
        p.append(Transform(f, [Identifiers.metacyc_reaction_id],
                           reaction_dbs_out + list(reaction_out.values()),
                           "Metacyc (X-id)"))

        # enzyme reaction metacyc id -> other id + other info (Enzyme Reaction)
        enzyme_dbs_out = []
        enzyme_out = {'ec-number': Identifiers.enzyme_EC,
                      'reaction': Identifiers.metacyc_reaction_id}
        # These might be available, but organisms are not mentioned so we ignore them.
        # 'kcat': 'Kinetics_kcat+', 'km': 'Kinetics_Km', 'vmax': 'Kinetics_Vmax'
        # Same here but for temperature/pH
        # 'equilibrium-constant': 'kinetics_Keq',
        f = template(Identifiers.metacyc_enzyme_reaction_id, enzyme_dbs_out, enzyme_out)
        p.append(Transform(f, [Identifiers.metacyc_enzyme_reaction_id],
                           enzyme_dbs_out + list(enzyme_out.values()),
                           "Metacyc (X-id)"))

        return p


class SabioRK(object):
    """Wrapper for the SabioRK REST API."""

    def __init__(self, shelve_cache=os.path.join(CACHE_DIR, "sabioRK.shelve"), delay=3):
        """Initialize the SabioRK REST API wrapper.

        Arguments:
        shelve_cache -- Shelve cache to save the result of previous queries.
                        If False, no cache will be used.
        delay -- Minimum delay between rest requests. If false, no delay will be used.
        """
        self.baseurl = ("http://sabiork.h-its.org/sabioRestWebServices/"
                        "searchKineticLaws/sbml?q=")
        self.shelve_cache = shelve_cache
        # Rate-limit the request so it doesn't spam the outside server.
        if delay > 0:
            self.request = rate_limited(delay)(self.request)
        # Wrap the rate-limited request in shelve cache if a location is provided.
        if self.shelve_cache:
            self.request = cached(shelve_cache)(self.request)

    def request(self, parameters):
        """Query the SabioRK database using the REST API.

        Arguments:
        parameters -- dictionary of key:parameter values. (Will be strung together
                      using AND)
        """
        # Parameters with spaces should have quotes.
        for i in parameters:
            if ' ' in parameters[i]:
                parameters[i] = '"{}"'.format(parameters[i])
        pstring = " AND ".join(str(k) + ':' + str(v) for k, v in parameters.items() if v)
        response = requests.get(self.baseurl+pstring)
        if response.ok:
            if response.text == u'No results found for query':
                return []
            else:
                return self.parse_response(response.text)
        elif response.status_code == 404:
            # No results, return empty list.
            return []
        else:
            # Others raise the error.
            response.raise_for_status()

    def parse_response(self, response):
        """Parse the rest SBML response into an easier dictionary based format.

        Arguments:
        response -- The SBML response from the sabioRK REST API.
        """
        # TODO: Sabio seems to have standard units, but does not document this anywhere?
        # Verify and add conversion if required...
        reaction_params = {'kcat': Identifiers.kinetics_kcat, 'Vmax': Identifiers.kinetics_vmax}
        compound_params = {'Km': Identifiers.kinetics_km, 'Ki': Identifiers.kinetics_ki,
                           'Ka': Identifiers.kinetics_ka, 'kcat_Km': Identifiers.kinetics_kcat_km}
        m = Model.from_string(response)
        reactions = []
        for reaction in m.reactions.values():
            r = {}
            r['id'] = reaction.id
            r['name'] = reaction.name
            r['identifiers'] = reaction.identifiers

            annotation = reaction.sbml_object.getKineticLaw().getAnnotationString()
            t = re.search(r'sbrk:startValueTemperature.([0-9]+\.[0-9]+).'
                          r'/sbrk:startValueTemperature', annotation)
            r['temperature'] = float(t.group(1)) if t is not None else None
            if r['temperature'] is not None and 0 > r['temperature'] > 100:
                # This is probably in Kelvin, so modify.
                r['temperature'] += 273.15
            pH = re.search(r'sbrk:startValuepH.([0-9]+\.[0-9]+).'
                           r'/sbrk:startValuepH', annotation)
            r['pH'] = float(pH.group(1)) if pH is not None else None

            r['reactants'] = {i.id: (i.name, i.identifiers) for i, _ in reaction.reactants}
            r['products'] = {i.id: (i.name, i.identifiers) for i, _ in reaction.products}
            # An extra check to prevent mutants enzyme modifiers.
            r['modifiers'] = {i.id: (i.name, i.identifiers) for i in reaction.modifiers
                              if 'mutant' not in i.name}
            pubmed = re.search(r'pubmed/([0-9]*)', annotation)
            r['pubmed'] = pubmed.group(1) if pubmed is not None else None

            buffers = re.search(r'sbrk:buffer.(.*?)./sbrk:buffer', annotation)
            r['comments'] = buffers.group(1) if buffers is not None else ''

            r['mutant'] = None
            for modifier in reaction.modifiers:
                if 'ENZ' in modifier.id:
                    r['comments'] = ' '.join((r['comments'], modifier.name))
                if 'mutant' in modifier.name:
                    r['mutant'] = modifier.name

            sabiork_ratelaw_id = re.search(r'<sbrk:kineticLawID>([0-9]+)</sbrk:kineticLawID>',
                                           annotation)
            if sabiork_ratelaw_id is not None:
                r[Identifiers.sabiork_ratelaw_id] = sabiork_ratelaw_id.group(1)
            else:
                r[Identifiers.sabiork_ratelaw_id] = None

            for p_id, p in reaction.parameters.items():
                if p_id in reaction_params:
                    r[reaction_params[p_id]] = (p['value'], m.getSIUnit(p['unit']))
                else:
                    try:
                        p_id, compound_id = p_id.split('_', 1)
                    except ValueError:
                        # These are compound parameters not identified with a compound.
                        # Not much we can do with them.
                        pass
                    else:
                        # Fix for kcat_Km
                        if p_id == 'kcat' and compound_id.startswith('Km'):
                                p_id_, compound_id = compound_id.split('_', 1)
                                p_id = p_id + '_' + p_id_
                        if p_id in compound_params:
                            if compound_params[p_id] not in r:
                                r[compound_params[p_id]] = []
                            r[compound_params[p_id]].append(((p['value'],
                                                              m.getSIUnit(p['unit'])),
                                                            compound_id))
            reactions.append(r)
        return reactions

    # def getTransforms(self):
    #     """Return all transforms using SabioRK data."""
    #     transforms = []

    #     def template(in_, out, reaction_parameters=True):
    #         """Template for reaction parameters (Vmax)."""
    #         # requestkeys_base = {'hasKineticData': 'true', 'IsRecombinant': 'false',
    #         #                     'EnzymeType': 'wildtype'}

    #         requestkeys_base = {}
    #         toAdd = [('Organism', 'organism_name')]

    #         for k, v in in_:
    #             # Sabio API key, transform key (Switch around)
    #             toAdd.append((v, k))

    #         def f(**kwargs):
    #             requestkeys = requestkeys_base.copy()
    #             for k, v in toAdd:
    #                 requestkeys[k] = kwargs[v]
    #             print out, requestkeys
    #             reactions = self.request(requestkeys)
    #             print reactions
    #             d = collections.defaultdict(list)
    #             for reaction in reactions:
    #                 # Add all the possible values that match.
    #                 for p in out:
    #                     if p in reaction:
    #                         if reaction_parameters:
    #                             # No need for compound matching.
    #                             d[p].append((reaction[p], None,
    #                                          reaction['temperature'], reaction['pH']))
    #                         else:
    #                             # Match compounds before adding.
    #                             # if ...:
    #                             # No need anymore since we do specific queries.
    #                             d[p].append(reaction[p](0), None, reaction['temperature'],
    #                                         reaction['pH'])
    #             print d
    #             if d:
    #                 exit()
    #             return d

    #         return f

    #     # For reaction parameters we only need a reaction identifier.
    #     out_reaction = ['Kinetics_Vmax+']
    #     in_reaction = {'kegg_reaction': 'KeggReactionID', 'sabiork_reaction': 'SabioReactionID'}
    #     # 'reaction': 'Enzymename'}

    #     for in_ in in_reaction.items():
    #         transforms.append(Transform(template([in_], out_reaction, True),
    #                           [in_[0]] + ['organism_name'], out_reaction))

    #     # For species we need both one of the species identifiers and one of the reaction ids.
    #     # Note: This will create a lot of small requests. Maybe it's better to request everything
    #     # for one species and parse it all at once
    #     out_species = ['Kinetics_kcat_km', 'Kinetics_Km', 'Kinetic_Ki',
    #                    'Kinetics_Ka', 'kinetics_kcat+']
    #     in_species = {'chebi_id': 'ChebiID', 'kegg_compound': 'KeggID',
    #                   'pubchem_id': 'PubChemID', 'species': 'AnyRole'}
    #     in_reaction = {'kegg_reaction': 'KeggReactionID', 'sabiork_reaction': 'SabioReactionID',
    #                    Identifiers.enzyme_EC: 'ECNumber'}
    #     for in_comb in itertools.product(in_species.items(), in_reaction.items()):
    #         transforms.append(Transform(template(in_comb, out_species, False),
    #                           [i[0] for i in in_comb] + ['organism_name'], out_species))

    #     return tuple(transforms)

    def getTransforms(self):
        """Return all transforms using SabioRK data."""
        # Note: To keep things faster and easier to parse, we only do requests
        # for reaction & organism combinations. Compounds are then parsed by hand.
        def template(in_, out):
            reaction_id = in_[0]
            species_id = in_[1]
            organism_id = in_[2]

            # Advertised but doesn't really work...
            # base = ['hasKineticData': 'true', 'IsRecombinant': 'false',
            #                     'EnzymeType': 'wildtype']
            base = []

            def f(**kwargs):
                # Retrieve kwargs
                org_reaction = reaction_id[1], kwargs[reaction_id[0]]
                organism = organism_id[1], kwargs[organism_id[0]]
                if species_id[0] is not None:
                    species = kwargs[species_id[0]]
                else:
                    species = None
                # Retrieve data
                query = {k: v for k, v in base + [org_reaction] + [organism]}
                reactions = self.request(query)

                d = collections.defaultdict(list)
                for reaction in reactions:
                    # Check for parameters
                    for p_type in out:
                        # Do we have this parameter?
                        if p_type in reaction:
                            # Do we need to check for specific species?
                            if (p_type == Identifiers.kinetics_kcat or
                                    p_type == Identifiers.kinetics_vmax):
                                reaction[p_type] = [(reaction[p_type], None)]
                            include = False
                            for (value, unit), owner in reaction[p_type]:
                                if (p_type == Identifiers.kinetics_kcat or
                                        p_type == Identifiers.kinetics_vmax):
                                    # We should add a check here if the reaction parameter
                                    # is actually relevant. While there is only one Vmax/kcat
                                    # for a reaction, it is still possible that the reaction
                                    # is not valid if identified on the basis of the enzyme,
                                    # since it might be functioning on a non standard metabolite?
                                    if species is None:
                                        include = True

                                else:
                                    # Check in identifiers or match name.
                                    where = None
                                    c = 0
                                    while where is None and c <= 2:
                                        if owner in reaction['products']:
                                            where = 'products'
                                        elif owner in reaction['reactants']:
                                            where = 'reactants'
                                        elif owner in reaction['modifiers']:
                                            where = 'modifiers'
                                        else:
                                            owner = owner.split('_', 1)[-1]
                                            c += 1
                                    if where is None:
                                        break

                                    if (((species_id[0], species) in reaction[where][owner][1]) or
                                            match_compounds(owner, species)):
                                        include = True

                                if include:
                                    std = None  # How to retrieve? Not in rest output
                                    source_query = "Sabio-Rk rate law: {} from query {}".format(
                                                    reaction[Identifiers.sabiork_ratelaw_id],
                                                    str(query))
                                    result = KineticResult(kinetics_value=value,
                                                           kinetics_type=p_type,
                                                           kinetics_std=std,
                                                           kinetics_unit=unit,
                                                           organism_name=kwargs[organism_id[0]],
                                                           mutant=reaction['mutant'],
                                                           kinetics_temp=reaction['temperature'],
                                                           kinetics_ph=reaction['pH'],
                                                           kinetics_comments=reaction['comments'],
                                                           species=kwargs.get(Identifiers.species),
                                                           reaction=kwargs.get(Identifiers.reaction),
                                                           source_name='Sabio-Rk',
                                                           source_query=source_query,
                                                           source_pubmed=reaction['pubmed'],
                                                           source_publication=None)
                                    result._sabio = reaction
                                    d[p_type].append(result)
                                    # Reset to False for next iteration
                                    include = False
                return dict(d)
            return f

        transforms = []
        in_organism = [(Identifiers.organism_name, 'Organism')]

        outReac = [Identifiers.kinetics_vmax, Identifiers.kinetics_kcat]
        outSpec = [Identifiers.kinetics_kcat_km, Identifiers.kinetics_km]
        outMod = [Identifiers.kinetics_ka, Identifiers.kinetics_ki]
        in_reaction = [(Identifiers.kegg_reaction, 'KeggReactionID'),
                       (Identifiers.enzyme_EC, 'ECNumber'),
                       (Identifiers.sabiork_reaction_id, 'SabioReactionID')]
        in_species = [(Identifiers.chebi_id, 'ChebiID'), (Identifiers.kegg_compound, 'KeggID'),
                      (Identifiers.pubchem_id, 'PubChemID'), (Identifiers.species, 'Substrate')]

        in_parameters = itertools.product(in_reaction, in_species, in_organism)
        basics = [Identifiers.species, Identifiers.reaction]

        for prod in in_parameters:
            f = template(prod, outSpec)
            transforms.append(Transform(f, [i[0] for i in prod] + basics,
                                        outSpec, "SabioRK (km/kcat_km)"))
            f = template(prod, outMod)
            transforms.append(Transform(f, [i[0] for i in prod] + basics,
                                        outMod, "SabioRK (ki/ka)"))

        in_parameters = itertools.product(in_reaction, [(None, None)], in_organism)

        for prod in in_parameters:
            f = template(prod, outReac)
            required = [prod[0][0], Identifiers.organism_name, Identifiers.reaction]
            transforms.append(Transform(f, required, outReac, "SabioRK (vmax/kv)"))

        return transforms


class MetaNetX(object):
    """Object to access MetaNetX tsv file data.

    For loading the data into the shelves, please look at `load_metanetx.py.`"""

    def __init__(self, shelve_path_prefix=os.path.join(CACHE_DIR, 'metanetx')):
        """Initialize the Metacyc data.

        Arguments:
        shelve_path_prefix -- file path prefix to metaNetX shelves. (For loading
                              the data into the shelves, please look at `load_metanetx.py.`)
        """
        self.chem_prop_shelve = shelve_path_prefix + "_chem_prop.shelve"
        self.reac_prop_shelve = shelve_path_prefix + "_reac_prop.shelve"
        self.reac_xref_shelve = shelve_path_prefix + "_reac_xref.shelve"
        self.chem_xref_shelve = shelve_path_prefix + "_chem_xref.shelve"
        self.names_shelve = shelve_path_prefix + "_names.shelve"

        if not self.checkfiles():
            raise MetaNetXError("Please generate the shelves first using"
                                "`load_metanetx.py`.")

    def checkfiles(self):
        """Check if all the shelve databases with metacyc output are available."""
        files = (self.chem_prop_shelve, self.reac_prop_shelve, self.reac_xref_shelve,
                 self.chem_xref_shelve, self.names_shelve)
        shelve = True
        for path in files:
            if not os.path.isfile(path):
                shelve = False
        return shelve

    def getTransforms(self):
        """Return all transforms using MetaNetX data."""
        transforms = []
        reaction_xrefs = [Identifiers.bigg_reaction_id, Identifiers.biopath_reaction_id,
                          Identifiers.kegg_reaction, Identifiers.metacyc_reaction_id,
                          Identifiers.reactome_reaction_id, Identifiers.rhea_id,
                          Identifiers.seed_reaction_id, Identifiers.unipathway_reaction_id]
        compound_xrefs = [Identifiers.bigg_metabolite_id, Identifiers.biopath_metabolite_id,
                          Identifiers.chebi_id, Identifiers.hmdb_id, Identifiers.inchi,
                          Identifiers.kegg_compound, Identifiers.lipid_maps_id,
                          Identifiers.metacyc_compound_id, Identifiers.reactome_metabolite_id,
                          Identifiers.seed_reaction_id, Identifiers.smiles,
                          Identifiers.umbbd_compound_id, Identifiers.unipathway_metabolite_id]

        # (name) -> (MetaNetX Compound)
        def f(**kwargs):
            db = shelve.open(self.names_shelve)
            # with shelve.open(self.names_shelve) as db:
            try:
                name = db[kwargs[Identifiers.species]]
            except KeyError:
                return {}
            else:
                return {Identifiers.metanetx_metabolite_id: name}
            finally:
                db.close()

        transforms.append(Transform(f, [Identifiers.species], [Identifiers.metanetx_metabolite_id],
                                    "MetaNetX (X-id)"))

        # (xref) -> (MetaNetX Compound)
        def template(id_in):
            def f(**kwargs):
                db = shelve.open(self.chem_xref_shelve)
                # with shelve.open(self.chem_xref_shelve) as db:
                try:
                    name = db[kwargs[id_in]]
                except KeyError:
                    return {}
                else:
                    return {Identifiers.metanetx_metabolite_id: name}
                finally:
                    db.close()
            return f

        for ref in compound_xrefs:
            transforms.append(Transform(template(ref), [ref], [Identifiers.metanetx_metabolite_id],
                                        "MetaNetX (X-id)"))

        # (MetaNetx compound) -> (xref)
        def f(**kwargs):
            db = shelve.open(self.chem_prop_shelve)
            try:
                data = db[Identifiers.metanetx_metabolite_id]
            except KeyError:
                return {}
            else:
                return dict(data['identifiers'])
            finally:
                db.close()

        transforms.append(Transform(f, [Identifiers.metanetx_metabolite_id], compound_xrefs,
                                    "MetaNetX (X-id)"))

        # (xref) -> (MetaNetX Reaction)
        def template(id_in):
            def f(**kwargs):
                db = shelve.open(self.reac_xref_shelve)
                try:
                    name = db[kwargs[id_in]]
                except KeyError:
                    return {}
                else:
                    return {Identifiers.metanetx_reaction_id: name}
                finally:
                    db.close()
            return f

        for ref in reaction_xrefs:
            transforms.append(Transform(template(ref), [ref], [Identifiers.metanetx_reaction_id],
                                        "MetaNetX (X-id)"))

        # (MetaNetX Reaction) -> (xref, E.C.)
        def f(**kwargs):
            db = shelve.open(self.reac_prop_shelve)
            try:
                data = db[Identifiers.metanetx_reaction_id]
            except KeyError:
                return {}
            else:
                identifiers = dict(data['identifiers'])
                try:
                    ec = data[Identifiers.enzyme_EC]
                except KeyError:
                    return identifiers
                else:
                    identifiers.update({Identifiers.enzyme_EC: ec})
                    return identifiers
            finally:
                db.close()

        transforms.append(Transform(f, [Identifiers.metanetx_reaction_id],
                                    reaction_xrefs + [Identifiers.enzyme_EC],
                                    "MetaNetX (X-id)"))
        return transforms
