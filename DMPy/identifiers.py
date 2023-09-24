#!/usr/bin/env python
"""
Scripts to reduce, parametrize and balance a biological model.

Namespace for some centralized identifier strings.


Author: Rik van Rosmalen
"""


class Identifiers(object):
    """Name space class for identifiers.

    Use these class variables to refer to in other scripts."""

    # Simple names
    organism_name = 'organism_name'
    reaction = 'reaction'
    species = 'species'
    mutant = 'mutant'

    # SBML IDs
    SBMLReactionID = 'SBMLReactionID'
    SBMLSpeciesID = 'SBMLSpeciesID'

    # References (Only for actual parameters, not for identifiers!)
    source = 'source'  # Text interpretation for reading
    source_url = 'source_url'  # Link to see resource
    source_name = 'source_name'  # Name of resource
    source_query = 'source_query'  # Text interpretation for reading
    source_publication = 'source_lit'  # Reference
    source_pubmed_id = 'source_pubmed_id'  # reference with Pubmed ID

    # Kinetic descriptors
    kinetics_type = 'kinetics_type'  # One of kinetics parameters types below
    kinetics_unit = 'kinetics_unit'  # Unit of a parameter
    kinetics_value = 'kinetics_value'  # Actual parameter value
    kinetics_std = 'kinetics_std'  # Parameter uncertainty as the stdev
    kinetics_temp = 'kinetics_temp'  # Temperature of a measurement (degree C)
    kinetics_ph = 'kinetics_ph'  # pH of a measurement
    kinetics_comments = 'kinetics_comments'  # Additional important comments

    # Kinetic parameters
    kinetics_ka = 'kinetics_Ka'  # Activation constant (mM)
    kinetics_keq = 'kinetics_Keq'  # Equilibrium constant (-)
    kinetics_ki = 'kinetics_Ki'  # Inhibition constant (mM)
    kinetics_km = 'kinetics_Km'  # Michaelis constant (mM)
    kinetics_kcat = 'kinetics_kcat+'  # Product catalytic rate constant (1/s)
    kinetics_kcat_km = 'kinetics_kcat_km'  # Kcat/Km
    kinetics_vmax = 'kinetic_Vmax+'  # Forward maximal velocity (mM/s)
    kinetics_kv = 'kinetics_kv'  # catalytic rate constant geometric mean (1/s)
    kinetics_ce = 'kinetics_ce'  # Enzyme concentration - mM
    kinetics_kcatm = 'kinetics_kcat-'  # Substrate catalytic rate constant (1/s)
    kinetics_vmaxm = 'kinetics_Vmax-'  # Backward maximal velocity (mM/s)
    kinetics_a = 'kinetics_A'  # Reaction affinity (kJ/mol)
    kinetics_mu = 'kinetics_mu'  # Chemical potential (kJ/mol)
    kinetics_mu0 = 'kinetics_mu0'  # Standard chemical potential (kJ/mol)
    kinetics_c = 'kinetics_c'  # Concentration (mM)

    # Some standards
    enzyme_EC = 'enzyme_EC'  # String of digits.digits.digits.digits
    inchi = 'inchi_id'  # InChI=[A-Za-z0-9\(\)\\\+\-,\?\/] (Not correct)
    inchi_key = 'inchi-key_id'  # InChIKey=[A-Z\-]+
    smiles = 'smiles_id'
    sbo = 'SBO'
    go = 'GO'

    # Assorted database identifiers
    # Reactions
    rhea_id = 'rhea_id'  # [0-9]{5}
    metanetx_reaction_id = 'metanetx_reaction_id'  # MNXR[0-9]+
    kegg_reaction = 'kegg_reaction'  # R[0-9]+
    sabiork_reaction_id = 'sabiork_reaction'  # [0-9]+
    sabiork_ratelaw_id = 'sabiork_ratelaw'  # [0-9]+
    metacyc_enzyme_reaction_id = 'metacyc_enzyme_reaction_id'
    metacyc_reaction_id = 'metacyc_reaction_id'
    # Metabolites
    chebi_id = 'chebi_id'  # [0-9]{5}
    kegg_compound = 'kegg_compound'  # C[0-9]+
    sabiork_metabolite_id = 'sabiork_metabolite'
    metacyc_compound_id = 'metacyc_compound_id'
    metanetx_metabolite_id = 'metanetx_metabolite_id'  # MNXM[0-9]+
    # Components
    metanetx_component_id = 'metanetx_component_id'  # MNXC[0-9]+

    # Some others that are found and can possibly be mapped.
    # Metabolites
    cas_id = 'cas_id'  # [0-9]{1,7}\-[0-9]{2}\-[0-9]
    chemspider_id = 'chemspider_id'  # [0-9]+
    drugbank_id = 'drugbank_id'  # DB[0-9]{5}
    ligand_cpd_id = 'ligand-cpd_id'  # Kegg compound
    lipid_maps_id = 'lipid_maps_id'  # LM(FA|GL|GP|SP|ST|PR|SL|PK)[0-9]{4}([0-9a-zA-Z]{4,6})?
    pubchem_id = 'pubchem_id'  # [0-9]+
    metabolights_id = 'metabolights_id'  # MTBLS[0-9]+
    umbbd_compound_id = 'umbbd_compund_id'  # c[0-9]+
    bigg_metabolite_id = 'bigg_metabolite_id'  # [a-z_A-Z0-9]+
    reactome_metabolite_id = 'reactome_metabolite'
    seed_metabolite_id = 'seed_metabolite_id'  # cpd[0-9]+
    biopath_metabolite_id = 'biopath_metabolite_id'
    unipathway_metabolite_id = 'unipathway_metabolite_id'
    # Reactions
    bigg_reaction_id = 'bigg_reaction_id'  # [a-z_A-Z0-9]+
    ecocyc_reaction_id = 'ecocyc_reaction_id'
    ligand_rxn_id = 'ligand-rxn_id'  # Kegg reaction
    reactome_reaction_id = 'reactome_reaction'  # ^(REACTOME:)?R-[A-Z]{3}-[0-9]+(-[0-9]+)? OR ^REACT_[0-9]+
    macie_id = 'macie'  # M[0-9]{4}
    seed_reaction_id = 'seed_reaction_id'  # cpd[0-9]+
    biopath_reaction_id = 'biopath_reaction_id'
    unipathway_reaction_id = 'unipathway_reaction_id'
    # Other / Multiple
    hmdb_id = 'hmdb_id'  # HMDB[0-9]{5}
    pir_id = 'pir_id'
    uniprot_id = 'uniprot_id'
    nci_id = 'nci_id'  # ???
