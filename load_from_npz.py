
"""
Data loading of npz file taken from https://github.com/spindro/AP-GCN/blob/0e75fd3af712a6a4e562ea52245065b6e6c4ef25/sparsegraph.py#L247
"""

import scipy.sparse as sp

sparse_graph_properties = [
        'adj_matrix', 'attr_matrix', 'labels',
        'node_names', 'attr_names', 'class_names',
        'metadata']

def from_flat_dict(data_dict) :
        """Initialize SparseGraph from a flat dictionary.
        """
        init_dict = {}
        del_entries = []

        # Construct sparse matrices
        for key in data_dict.keys():
            if key.endswith('_data') or key.endswith('.data'):
                if key.endswith('_data'):
                    sep = '_'
                else:
                    sep = '.'
                matrix_name = key[:-5]
                mat_data = key
                mat_indices = '{}{}indices'.format(matrix_name, sep)
                mat_indptr = '{}{}indptr'.format(matrix_name, sep)
                mat_shape = '{}{}shape'.format(matrix_name, sep)
                if matrix_name == 'adj' or matrix_name == 'attr':

                    matrix_name += '_matrix'
                
                init_dict[matrix_name] = sp.csr_matrix(
                        (data_dict[mat_data],
                         data_dict[mat_indices],
                         data_dict[mat_indptr]),
                        shape=data_dict[mat_shape])
                del_entries.extend([mat_data, mat_indices, mat_indptr, mat_shape])

        # Load everything else
        for key, val in data_dict.items():
            if ((val is not None) and (None not in val)):
                init_dict[key] = val

        return init_dict