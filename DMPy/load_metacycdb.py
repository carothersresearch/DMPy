#!/usr/bin/env python
"""
Scripts to load the flat files from metacyc into a more accessible shelve db.

Creates multiple files:
- Database accessed by metacyc id (Compounds, Reactions and Enzyme Reactions)
- 3 separate databases accessed by names linking to metacyc id.
- Database with all cross-identifiers linking to metacyc id.

Author: Rik van Rosmalen
"""
from __future__ import division
from __future__ import print_function

import re
import collections
import shelve

# For a more complete list of tags, check the metacyc flat file headers.
name_tags = set(('COMMON-NAME', 'ABBREV-NAME', 'SYNONYMS', 'SYSTEMATIC-NAME'))
identifier_tags = set(('DBLINKS', 'SMILES', 'INCHI', 'INCHI-KEY', 'REACTION',
                       'ENZYMATIC-REACTION'))
include_tags = set(('CATALYZES', 'COFACTORS-OF', 'COMPONENT-COEFFICIENTS',
                    'COMPONENT-OF', 'COMPONENTS', 'DATA-SOURCE', 'REACTION',
                    'GENE', 'GO-TERMS', 'INCHI', 'INCHI-KEY', 'SMILES', 'SYNONYMS',
                    'SYSTEMATIC-NAME', 'ALTERNATIVE-COFACTORS', 'ALTERNATIVE-SUBSTRATES',
                    'COFACTORS', 'ENZRXN-IN-PATHWAY', 'ENZYME', 'KCAT', 'KM',
                    'REACTION-DIRECTION', 'REGULATED-BY', 'SPECIFIC-ACTIVITY', 'VMAX',
                    'EC-NUMBER', 'ENZYMATIC-REACTION', 'EQUILIBRIUM-CONSTANT', 'LEFT',
                    'RIGHT', 'REACTION-LIST', 'STD-REDUCTION-POTENTIAL'
                    ))


def blocks(iterator, nextid='//', includenext=False, comment=('#')):
    """Take an iterator and returns blocks of lines separated by nextid."""
    temp = []
    for line in iterator:
        line = line.strip()
        if line.startswith(comment):
            continue
        # Hack for metacyc files. Sometimes the comment can be multiple lines
        # starting with a single '/'
        elif line.startswith('/') and not line.startswith(nextid):
            continue
        elif nextid in line:
            yield temp
            if includenext:
                temp = [line]
            else:
                temp = []
        else:
            temp.append(line)
    else:
        raise StopIteration


def expandname(name):
    """Expand a compound name into other logical possibilities.

    For example, remove inline markup or replace special characters.
    """
    # Remove <sup> or <sub> or <i> tags etc.
    n1 = re.sub(r'<(.*?)>', '', name)
    # Replace &alpha; type substrings with just the content
    n2 = re.sub(r'&(.*?);', r'\g<1>', name)
    # Replace &alpha; type substrings with just the first letter
    # (Hack which is fine for alhpa and beta...)
    n3 = re.sub(r'&(.).*?;', r'\g<1>', name)

    # Remove both for both options
    # Can still be improved with some combinatorics...
    n4 = re.sub(r'&(.*?);', r'\g<1>', n1)
    n5 = re.sub(r'&(.).*?;', r'\g<1>', n1)

    return set((n1, n2, n3, n4, n5))


def getData(filename):
    """Collect the data from a metacyc flatfile."""
    # TODO: Fix enzymatic reaction links. One reaction can have multiple enzymatic reactions.
    identifiers = collections.defaultdict(dict)
    all_names = {}
    all_data = {}

    with open(filename, 'r') as infile:
        for block in blocks(infile):
            names = set()

            data = {}

            for line in block:
                try:
                    tag, value = (i.strip() for i in line.split(' - ', 1))
                except ValueError:
                    pass  # Empty tag, so we couldn't split.

                if tag == "UNIQUE-ID":
                    id_ = value

                # Add all possible names for lookups.
                if tag in name_tags:
                    names.add(value.lower())

                # Identifiers for looking it up by identifier.
                if tag in identifier_tags:
                    if tag == 'DBLINKS':
                        # Remove parantheses
                        value = value[1:-1]
                        value = value.split(' ')
                        # If there is any other relation then NIL, we're not insterested
                        # for lookup purposes.
                        if len(value) >= 3 and (value[2] == 'RELATED-TO' or value[2] == 'PART'):
                            continue
                        db = value[0]
                        value = value[1].strip()
                        # Check for and remove quotes
                        if value[0] == value[-1] and value[0] in ('"', ","):
                            value = value[1:-1]
                        data[db.lower()] = value
                        identifiers[db.lower()][value] = id_
                    else:
                        value = value.strip()
                        # Check for and remove quotes
                        if value[0] == value[-1] and value[0] in ('"', ","):
                            value = value[1:-1]
                        data[tag.lower()] = value
                        identifiers[tag.lower().strip()][value] = id_

                # Only save the tags on the include list
                if tag in include_tags:
                    if tag == 'EC-NUMBER':
                        value = value[3:]
                    if tag.lower() in data:
                        # Make a list or if it already exists append.
                        try:
                            data[tag.lower()].append(value)
                        except AttributeError:
                            data[tag.lower()] = [data[tag.lower()], value]

                    data[tag.lower()] = value

            # All tags have been scanned, now add to necessary variables.
            # For lookup by name
            for name in names.copy():
                names |= expandname(name)
            for name in names:
                all_names[name] = id_

            # All other data can be looked up by ID.
            all_data[id_] = data

    return all_data, all_names, identifiers


def writetoshelve(data, shelve_path):
    """Write a dictionary to a shelve per key."""
    if '.shelve' not in shelve_path:
        shelve_path += '.shelve'
    s = shelve.open(shelve_path, protocol=-1)
    for key, value in data.items():
        # time.sleep(1)  # Bugfix, see: http://stackoverflow.com/a/12167172
        s[key] = value
    s.close()


def addname(name, dict_):
    """Add name in front of all the keys in dict_."""
    return {name + key: value for key, value in dict_.items()}


def main(cfile, efile, rfile, shelve_path):
    """Collect all data, and file away in shelves."""
    c = getData(cfile)
    e = getData(efile)
    r = getData(rfile)

    # names = 'metacyc', 'names', 'idenfifiers'

    # for name, data in zip(names, zip(c, e, r)):
    #     # Test for clashes
    #     print "Clashes", name
    #     print len(set(data[1].keys() + data[2].keys()) & set(data[0].keys()))
    #     print len(set(data[2].keys() + data[0].keys()) & set(data[1].keys()))
    #     print len(set(data[0].keys() + data[1].keys()) & set(data[2].keys()))

    combined = {}
    for i in c[0], e[0], r[0]:
        combined.update(i)

    # All data (combined since identifiers are unique anyway)
    writetoshelve(combined, shelve_path + '_data')
    # Compound names
    writetoshelve(c[1], shelve_path + '_names_compounds')
    # Reaction names
    writetoshelve(r[1], shelve_path + '_names_reaction')
    # Enzyme reaction names
    writetoshelve(e[1], shelve_path + '_names_enzyme_reaction')
    # Identifiers
    c[2].update(e[2])
    c[2].update(r[2])
    i = c[2]
    writetoshelve(c[2], shelve_path + '_identifiers')

    return c, e, r, combined, i


if __name__ == "__main__":
    cfile = 'data/local/compounds.dat'
    efile = 'data/local/enzrxns.dat'
    rfile = 'data/local/reactions.dat'
    shelve_path = 'shelve_cache/metacyc'
    c, e, r, combined, i = main(cfile, efile, rfile, shelve_path)
