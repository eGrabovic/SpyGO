from stl import mesh

# Load the STL file

stl_file = 'example.stl'

mesh_data = mesh.Mesh.from_file(stl_file)

# Accessing mesh information

print('Number of facets:', len(mesh_data.x))

print('Facet normals:', mesh_data.normals)