import argparse
import numpy as np
import os.path
import cupy as cp
import sys
import datetime
import os
import psutil
import sparse
import operator

from scipy.sparse import coo_matrix

from dask.distributed import Client
from dask.bag import read_text
import dask.array as da

def row_to_numpy(row):

    return np.array([np.float(x) for x in row.split()])

def load_data(path, chunks):

    raw_bag = read_text(path) \
                .str.strip() \
                .map(row_to_numpy)

    return da.stack(raw_bag,axis=0).map_blocks(cp.array)

def normalize(dask_array):

    dask_array -= dask_array.mean()
    dask_array /= da.linalg.norm(dask_array)
    return dask_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Dask Dictionary Learning',
        add_help = 'How to use', prog = 'python R1DL_DASK.py <args>')

    # Inputs.
    parser.add_argument("-i", "--input", required = True,
        help = "Input file containing the matrix S.")
    parser.add_argument("-T", "--rows", type = int, required = True,
        help = "Number of rows (observations) in the input matrix S.")
    parser.add_argument("-P", "--cols", type = int, required = True,
        help = "Number of columns (features) in the input matrix S.")

    # Optional.
    parser.add_argument("-r", "--pnonzero", type = float, default = 0.07,
        help = "Percentage of non-zero elements. [DEFAULT: 0.07]")
    parser.add_argument("-m", "--dictatoms", type = int, default = 5,
        help = "Number of the dictionary atoms. [DEFAULT: 5]")
    parser.add_argument("-e", "--epsilon", type = float, default = 0.01,
        help = "The convergence criteria in the ALS step. [DEFAULT: 0.01]")
    parser.add_argument("--normalize", action = "store_true",
        help = "If set, normalizes input data.")
    parser.add_argument("--debug", action = "store_true",
        help = "If set, turns out debug output.")

    # Dask options.
    parser.add_argument("--chunks", action = "store_true",
        help = "If set, then chucks will be (1,cols). If not, chucks is 'auto'")
    #parser.add_argument("--execmem", default = "8g",
    #    help = "Amount of memory for each executor. [DEFAULT: 8g]")

    # Outputs.
    parser.add_argument("-d", "--dictionary", required = True,
        help = "Output path to dictionary file.(file_D)")
    parser.add_argument("-o", "--output", required = True,
        help = "Output path to z matrix.(file_z)")
    parser.add_argument("--prefix", required = True,
        help = "Prefix strings to the output files")

    args = vars(parser.parse_args())

    if args['debug']: print(datetime.datetime.now())

    #Add Comment
    client = Client()

    #Ensuring each chunk of data be atleast one full row will make row and column
    #wise operations easier. Hasnt been used yet.....
    chunks = (1, args['cols']) if args['chunks'] is True else 'auto'

    # Read the data and convert it into a Dask Array.
    S = load_data(args['input'], chunks)
    if args['normalize']:
        S = normalize(S)


    ##################################################################
    # Here's where the real fun begins.
    #
    # First, we're going to initialize some variables we'll need for the
    # following operations. Next, we'll start the optimization loops. Finally,
    # we'll perform the stepping and deflation operations until convergence.
    #
    # Sound like fun?
    ##################################################################

    T = args['rows']
    P = args['cols']

    epsilon = args['epsilon']       # convergence stopping criterion
    M = args['dictatoms']            # dimensionality of the learned dictionary
    R = args['pnonzero'] * P        # enforces sparsity
    u_new = da.zeros(T)
    v = da.zeros(P)

    max_iterations = P * 10
    file_D = os.path.join(args['dictionary'], "{}_D.txt".format(args["prefix"]))
    file_z = os.path.join(args['output'], "{}_z.txt".format(args["prefix"]))

    # Start the loop!
    for m in range(M):
        #Let us randomly generate a integer, broadcast that int, and create a seed.
        seed = np.random.randint(max_iterations + 1, high = 4294967295)
        _SEED_ = client.scatter(seed, broadcast=True)
        np.random.seed(seed)

        #Create a dense random vector
        #Then subtracting off the mean an normalizing it
        u_old = da.random.random(T)
        u_old = normalize(u_old)

        #Setting loop criteria
        num_iterations = 0
        delta = 2 * epsilon

        # Start the inner loop: this learns a single atom.
        while num_iterations < max_iterations and delta > epsilon:
            _U_ = client.scatter(u_old, broadcast=True)
            v = da.matmul(_U_.result(),S)

            #Grab the indices and data of the top R values in v for the sparse vector
            indices = v.argtopk(R,axis=0)
            data = v[indices].compute()

            print('making the sparse vector')
            #let's make the sparse vector.
            sv = sparse.COO(indices,data,shape=(P),sorted=False)
            sv = da.from_array(sv)
            print('made the sparse vector')
            # Broadcast the sparse vector.
            _V_ = client.scatter(sv,broadcast=True)

            # P1: Matrix-vector multiplication step. Computes u.
            u_new = da.matmul(S,_V_.result())

            # Subtract off the mean and normalize.
            u_new = normalize(u_new).compute()

            # Update for the next iteration.
            delta = da.linalg.norm(u_old - u_new) #Should u_old be _U_?
            u_old = u_new
            num_iterations += 1

        # Save the newly-computed u and v to the output files;
        with open(file_D, "a+") as fD:
            np.savetxt(fD, u_new, fmt = "%.6f", newline = " ")
            fD.write("\n")
        with open(file_z, "a+") as fz:
            np.savetxt(fz, sv.compute().todense(), fmt = "%.6f", newline = " ")
            fz.write("\n")


        # P4: Deflation step. Update the primary data matrix S.
        _U_ = client.scatter(u_new, broadcast=True)
        _V_ = client.scatter(sv, broadcast=True)


        if args['debug']: print(m)

        S -= da.core.blockwise(operator.mul, 'ij', _U_.result(), 'i', _V_.result(), 'j', dtype='f64')
        S.persist()

    if args['debug']: print(datetime.datetime.now())
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)
