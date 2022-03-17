import numpy as np

a = np.random.random(size=(6, 5, 5))
b = np.random.random(size=(6, 1))

with open('test.txt', 'w') as outfile:
    outfile.write('# Array shape: {0}\n'.format(a.shape))
    for slice_2d in a:
        outfile.write('# New slice\n')
        np.savetxt(outfile, slice_2d)
    outfile.write('# bias:\n')
    np.savetxt(outfile,b)
