import numpy as np
import os

files = os.listdir('.')
files = [f for f in files if f.split('.')[-1] == 'dat' and 'centered' not in f]

for filename in files:
    print("Processing " + filename)

    f = open(filename, 'r')

    mk = f.readline().strip().split()
    m = int(mk[0])
    k = int(mk[1])

    mu_x = f.readline().strip().split()
    mu_y = f.readline().strip().split()
    mu_z = f.readline().strip().split()

    mu_x = np.array([float(x) for x in mu_x])
    mu_y = np.array([float(y) for y in mu_y])
    mu_z = np.array([float(z) for z in mu_z])

    mu = np.stack((mu_x, mu_y, mu_z))

    center = mu.mean(1)

    centered_mu = mu - center.reshape((3,1))

    lines = []
    lines += ['{:d} {:d}\n'.format(m,k)]
    lines += [' '.join([str(x) for x in centered_mu[0,:].tolist()]) + '\n']
    lines += [' '.join([str(x) for x in centered_mu[1,:].tolist()]) + '\n']
    lines += [' '.join([str(x) for x in centered_mu[2,:].tolist()]) + '\n']

    # now for each deformation basis...
    for i in range(k):
        Bk_x = f.readline().strip().split()
        Bk_y = f.readline().strip().split()
        Bk_z = f.readline().strip().split()

        Bk_x = np.array([float(x) for x in Bk_x])
        Bk_y = np.array([float(y) for y in Bk_y])
        Bk_z = np.array([float(z) for z in Bk_z])

        centered_Bk = np.stack((Bk_x, Bk_y, Bk_z)) - center.reshape((3,1))

        lines += [' '.join([str(x) for x in centered_Bk[0,:].tolist()]) + '\n']
        lines += [' '.join([str(x) for x in centered_Bk[1,:].tolist()]) + '\n']
        lines += [' '.join([str(x) for x in centered_Bk[2,:].tolist()]) + '\n']


    f.close()

    # new_name = '{}_centered.dat'.format(filename.split('.')[0])

    with open(filename, 'w') as f:
        f.writelines(lines)