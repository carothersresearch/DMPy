#!/usr/bin/env python
"""
"Script to load up the MetaNetX data into a shelve database for use in the pipeline.

Creates multiple files, see bottom.

Author: Rik van Rosmalen
"""
from __future__ import division
from __future__ import print_function

import shelve

from identifiers import Identifiers

chem_prop = 'data/local/chem_prop.tsv'
chem_xref = 'data/local/chem_xref.tsv'
reac_prop = 'data/local/reac_prop.tsv'
reac_xref = 'data/local/reac_xref.tsv'

id_dict = {
    'chebi': Identifiers.chebi_id,
    'kegg': (Identifiers.kegg_reaction, Identifiers.kegg_compound),
    'metacyc': (Identifiers.metacyc_reaction_id, Identifiers.metacyc_compound_id),
    'upa': (Identifiers.unipathway_reaction_id, Identifiers.unipathway_metabolite_id),
    'seed': (Identifiers.seed_reaction_id, Identifiers.seed_reaction_id),
    'bigg': (Identifiers.bigg_reaction_id, Identifiers.bigg_metabolite_id),
    'biopath': (Identifiers.biopath_reaction_id, Identifiers.biopath_metabolite_id),
    'lipidmaps': Identifiers.lipid_maps_id,
    'hmdb': Identifiers.hmdb_id,
    'reactome': (Identifiers.reactome_reaction_id, Identifiers.reactome_metabolite_id),
    'umbbd': Identifiers.umbbd_compound_id,
    'MNXR': Identifiers.metanetx_reaction_id,
    'MNXC': Identifiers.metanetx_metabolite_id,
    'rhea': Identifiers.rhea_id,
    'mnx': (Identifiers.metanetx_reaction_id, Identifiers.metanetx_metabolite_id),
    'sabiork': (Identifiers.sabiork_reaction_id, Identifiers.sabiork_metabolite_id),
    'envipath': 'envipath',  # Not supported yet
    'slm': 'slm',  # Not supported yet
    }


def readfile(filename, sep='\t', force_key=False):
    """Read MetaNetX tsv file and return a generator with the header and the results."""
    # Find header
    with open(filename, 'r') as f:
        oldline = '#'
        for line in f:
            if not line.startswith('#'):
                header = oldline
                header = header.strip('#').strip().split(sep)
                break
            else:
                oldline = line
    yield header
    with open(filename, 'r') as f:
        for line_nr, line in enumerate(f):
            if line.startswith('#'):
                continue
            else:
                line = line.strip()
                if not line:
                    continue
                line = line.split(sep)
                new = []
                skip = False
                for i, value in enumerate(line):
                    if i >= len(header):
                        # print("Value out of header range ({}; line {}): {}".format(filename, line_nr, value))
                        continue
                    elif ':' in value and header[i] in {'Source', 'XREF'}:
                        key, value = value.split(':', 1)
                        if key == 'deprecated':
                            skip = True
                        new.append((key, value))
                    elif force_key and header[i] in {'Source', 'XREF'}:
                        new.append((force_key, value))
                    else:
                        new.append(value)
                # Pad values if we are shorter then the header.
                while len(new) < len(header):
                    new.append('')
                if skip:
                    continue
                if len(new) == 3 and new[2] == 'identity':
                    print("Skipped empty identity ({}; line {}): {}".format(filename, line_nr, new))
                    continue
                yield new


def writetoshelve(data, shelve_path):
    """Write a dictionary to a shelve per key."""
    if '.shelve' not in shelve_path:
        shelve_path += '.shelve'
    s = shelve.open(shelve_path, protocol=-1)
    for key, value in data.items():
        s[key] = value
    s.close()


chem_xref_data = {}
chem_prop_data = {}
reac_xref_data = {}
reac_prop_data = {}

name_to_MNXM = {}

# Metabolite cross references
lines = readfile(chem_xref, force_key='MNXC')
header = lines.next()
for line in lines:
    (id_, key), MNXM, _, names = line
    # Split names if we have multiple
    if '|' in names:
        names = names.split('|')
    else:
        names = [names]
    # Convert id to correct name
    id_ = id_dict[id_]
    if isinstance(id_, tuple):
        id_ = id_[1]
    # Skip these
    if id_ in ('envipath', 'slm', Identifiers.metanetx_metabolite_id):
        continue
    # Save data
    if (id_, key) in chem_xref_data:
        print("Duplicate xref:", (id_, key), MNXM)
    chem_xref_data[(id_, key)] = MNXM
    for name in names:
        name_to_MNXM[name] = MNXM

# Metabolite properties
lines = readfile(chem_prop)
header = lines.next()
for line in lines:
    MNXM, name, formula, charge, mass, inchi, smiles, source, inchikey = line
    try:
        source_id, source_key = source
    except ValueError:
        if MNXM == source:
            print("Skipping self reference: {} -> {}".format(MNXM, source))
        continue
    # Covert source id to correct name
    source_id = id_dict[source_id]
    if isinstance(source_id, tuple):
        source_id = source_id[1]
    # Collect data
    data = {'identifiers': [(source_id, source_key),
                            (Identifiers.inchi, inchi),
                            (Identifiers.smiles, smiles),
                            (Identifiers.inchi_key, inchikey)
                            ],
            'name': name,
            'formula': formula,
            'charge': charge,
            'mass': mass,
            }
    # Save data
    chem_prop_data[MNXM] = data
    if name not in name_to_MNXM:
        name_to_MNXM[name] = MNXM

# Update crossrefs in properties
for (db, key), MNXC in chem_xref_data.items():
    if MNXC not in chem_prop_data:
        print(MNXC, "not found in data")
        continue
    ids = chem_prop_data[MNXC]['identifiers']
    if (db, key) not in ids:
        ids.append((db, key))

# Update crossrefs with smiles and inchi
for MNXC, data in chem_prop_data.items():
    for (db, key) in data['identifiers']:
        if db in {Identifiers.inchi, Identifiers.smiles, Identifiers.inchi_key}:
            chem_xref_data[(db, key)] = MNXC

# Reaction cross references
lines = readfile(reac_xref, force_key='MNXR')
header = lines.next()
for line in lines:
    (id_, key), MNXR = line
    # Convert id to correct name
    id_ = id_dict[id_]
    if isinstance(id_, tuple):
        id_ = id_[0]
    if id_ in (Identifiers.metanetx_reaction_id):
        continue
    # Save data
    if (id_, key) in reac_xref_data:
        print("Duplicate xref:", (id_, key), MNXR)
    reac_xref_data[(id_, key)] = MNXR

# Reaction properties
lines = readfile(reac_prop)
header = lines.next()
for line in lines:
    MNXR, eq, eq_desc, balanced, ec, source = line
    try:
        source_id, source_key = source
    except ValueError:
        print("Skipping", MNXR, source)
        continue
    # Covert source id to correct name
    source_id = id_dict[source_id]
    if isinstance(source_id, tuple):
        source_id = source_id[1]
    # Check for multiple ec numbers.
    if ';' in ec:
        ec = ec.split(';')
    data = {'identifiers': [(source_id, source_key)],
            'equitation': eq,
            'equitation_description': eq_desc,
            Identifiers.enzyme_EC: ec
            }
    reac_prop_data[MNXR] = data

# Update crossrefs in properties
for (db, key), MNXR in reac_xref_data.items():
    if MNXR not in reac_prop_data:
        print(MNXR, "not found in data")
        continue
    ids = reac_prop_data[MNXR]['identifiers']
    if (db, key) not in ids:
        ids.append((db, key))

dbs = [(chem_xref_data, 'shelve_cache/metanetx_chem_xref.shelve'),
       (chem_prop_data, 'shelve_cache/metanetx_chem_prop.shelve'),
       (reac_xref_data, 'shelve_cache/metanetx_reac_xref.shelve'),
       (reac_prop_data, 'shelve_cache/metanetx_reac_prop.shelve'),
       (name_to_MNXM, 'shelve_cache/metanetx_names.shelve'),
       ]

for data, path in dbs:
    print("Saving:", path)
    writetoshelve({str(key): value for key, value in data.items()}, path)
