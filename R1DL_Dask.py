import argparse
import numpy as np
import os.path
import scipy.linalg as sla
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
    parser.add_argument("--chunks", type = int, default = 'auto',
        help = "Number of RDD partitions to use. [DEFAULT: 4 * CPUs]")
    parser.add_argument("--execmem", default = "8g",
        help = "Amount of memory for each executor. [DEFAULT: 8g]")

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
    conf.set("spark.executor.memory", args['execmem'])
    
    chunks = args['chunks'] if args['chunks'] is not None else (4 * sc.defaultParallelism)

    # Read the data and convert it into a Dask Array.
    raw_data = db.read_text(args['input'])
    S = da.from_array(raw_data, chunks=chunks)
    if args['normalize']:
        S -= S.mean()
        S /= sla.norm(S)


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
        #Add a note here. But ask Quinn if this is the best thing to do
        seed = np.random.randint(max_iterations + 1, high = 4294967295)
        _SEED_ = client.scatter(seed, broadcast=True)
        np.random.seed(seed)

        #Create a dense random vector
        #Then subtracting off the mean an normalizing it
        u_old = da.random.random(T)
        u_old -= u_old.mean()
        u_old /= sla.norm(u_old, axis = 0)

        #Setting loop criteria
        num_iterations = 0
        delta = 2 * epsilon

        # Start the inner loop: this learns a single atom.
        while num_iterations < max_iterations and delta > epsilon:
            _U_ = client.scatter(u_old, broadcast=True)
            v = da.dot(_U_.result(),S) #May get an error here because S may be a future instead of a dask array

            #Grab the indices and data of the top R values in v for the sparse vector
            indices = np.sort(v.argtopk(R,axis=0))
            data = v[indices].compute()  #Do I need to delete any of these intermediate variables?

            #let's make the sparse vector.
            sv = sparse.COO(indices,data,shape=(P),sorted=True)
            sv = da.from_array(sv)

            # Broadcast the sparse vector.
            _V_ = client.scatter(sv,broadcast=True)

            # P1: Matrix-vector multiplication step. Computes u.
            u_new = da.dot(S,_V_.result())


            # Subtract off the mean and normalize.
            u_new -= u_new.mean()
            u_new /= sla.norm(u_new,axis = 0)
            u_new = u_new.compute()

            # Update for the next iteration.
            delta = sla.norm(u_old - u_new) #Should u_old be _U_?
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

        S -= da.core.blockwise(operator.mul, 'ij', _U_.result(), 'i', _V_.result(), 'j', dtype='f8')
        S.persist()

    if args['debug']: print(datetime.datetime.now())
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)
