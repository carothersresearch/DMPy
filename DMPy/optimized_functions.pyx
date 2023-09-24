# !-! cython: linetrace=True
# !-! distutils: define_macros=CYTHON_TRACE_NOGIL=1

def build_dependency_matrix(data_in, stoichiometry, dependencies,
                            c_pos, r_pos, columns, Q):
    for row, (_, _, parameter, compound, reaction) in enumerate(data_in):
        for column, (dependent_p, dependent_c, dependent_r) in enumerate(columns):
            # Is this dependent and part of the right compound/reaction combo?
            if dependent_p in dependencies[parameter]:
                d, multiply = dependencies[parameter][dependent_p]
            else:
                continue

            if (((dependent_c == compound and dependent_r == reaction) or
                 (dependent_c == compound and reaction is None) or
                 (compound is None and dependent_r == reaction))):
                Q[row, column] = d

            if multiply:  # Multiply by stoichiometry?
                x = c_pos[compound if compound is not None else dependent_c]
                y = r_pos[reaction if reaction is not None else dependent_r]
                m = stoichiometry[x, y]
                if m:
                    if not Q[row, column]:  # Still need to set the base term!
                        Q[row, column] = d
                    Q[row, column] *= m
    return Q
