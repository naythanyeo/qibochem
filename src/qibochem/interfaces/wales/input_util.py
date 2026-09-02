import numpy as np
from scipy.sparse import csc_matrix, save_npz
from pathlib import Path
import openfermion
from openfermion.linalg import get_sparse_operator
from qibochem.driver.hamiltonian import _fermionic_hamiltonian, _qubit_hamiltonian
from qibochem.interfaces.wales.fortran_helpers import fortran_float

# save hamiltoninan
def gen_hamiltonian(obj, path=Path.cwd()):
    hamiltonian_file_path = path / "qc_ucc_hamiltonian.dat"
    index_file_path = path / "qc_ucc_index.dat"
    try: # check if perfect_pair attribute exists, if not set to False
        perfect_pair = obj.perfect_pair
    except:
        perfect_pair = False
    if obj.perfect_pair:
        oei = obj.mol.embed_oei
        tei = obj.mol.embed_tei
        oei_pp = oei[np.ix_(obj.mo_perm, obj.mo_perm)]
        tei_pp = tei[np.ix_(obj.mo_perm, obj.mo_perm, obj.mo_perm, obj.mo_perm)]
        ferm_op = obj.mol.hamiltonian("ferm", oei=oei_pp, tei=tei_pp)
    else:
        ferm_op = obj.mol.hamiltonian("ferm")
    blocked_ferm_op = openfermion.transforms.reorder(ferm_op, openfermion.utils.up_then_down, num_modes=obj.n_active_spin)
    blocked_h_mat = get_sparse_operator(openfermion.transforms.jordan_wigner(blocked_ferm_op), n_qubits=obj.n_active_spin)


    rows, columns = blocked_h_mat.nonzero()
    values = np.asarray(blocked_h_mat[rows, columns]).ravel().real

    np.savetxt(hamiltonian_file_path, values, fmt="%.20f")

    np.savetxt(index_file_path, np.column_stack((rows + 1, columns + 1)), fmt="%d")

# save the null vector
def gen_coords(obj, path=Path.cwd()):
    file_path = path / "coords"
    np.savetxt(file_path, np.zeros(len(obj.param_names)), delimiter="\n", fmt='%0.0f')

# save op order
def gen_order(obj, path=Path.cwd()):
    order_list = []
    file_path = path / "qc_ucc_order.dat"
    for idx, name in enumerate(obj.param_names):
        order_list.append(idx+1)
        if 's' in name: # if spin-adapted singles append order list with the same number
            order_list.append(idx+1)
    order_array = np.array(order_list)
    np.savetxt(file_path, order_array, delimiter="\n", fmt='%0.0f')

# save orbital excitation ordering:
def gen_excitations(obj, path=Path.cwd()):
    file_path = path / "qc_ucc.dat"
    open(file_path,'w').close()
    with open(file_path,'a') as f:
        for name in obj.param_names:
            for weight, ex in obj.param_excitations[name]:
                if len(ex[0]) == 1:
                    p = ex[1][0] // 2 + 1
                    q = ex[0][0] // 2 + 1
                    if ex[0][0] % 2 == 0:
                        excitation = np.array([q,p,0,0])
                    else:
                        excitation = np.array([q+obj.n_active_spat,p+obj.n_active_spat,0,0])
                elif len(ex[0]) == 2:
                    p = ex[1][0] // 2 + 1
                    q = ex[0][0] // 2 + 1
                    excitation = np.array([q,q+obj.n_active_spat,p,p+obj.n_active_spat])


                np.savetxt(f,excitation[None,:], fmt='%0.0f')

# save initial bitstring by converting the reference bitstring to a permuted bitstring. ie. 11001100 -> 10101010
def gen_initial(obj, path=Path.cwd()):
    file_path = path / "qc_ucc_initial.dat"
    permuted_bitstring = obj.ref_bitstring[::2] + obj.ref_bitstring[1::2]
    with open(file_path, 'w') as f:
        f.write(permuted_bitstring[::-1])

        

# save data file
def gen_data(obj, path=Path.cwd(), temp=1, 
             sloppyconv=1e-6, tightconv=1e-7, 
             save=10000, ediff=1e-6, updates=10, 
             maxerise=1e-6, maxit=100000, 
             step=(1.0, 0.5), steps=10000, tfac=1.0):
    file_path = path / "data"
    with open(file_path, 'w') as f:
        f.write(
f'''! RADIUS 1.0D10
! CHECKD
SLOPPYCONV {fortran_float(sloppyconv)}
TIGHTCONV {fortran_float(tightconv)}
SAVE {save}
EDIFF {fortran_float(ediff)}
UPDATES {updates}
MAXERISE {fortran_float(maxerise)}
MAXIT {maxit} {maxit}
STEPS {steps} {tfac}
TEMPERATURE {temp}
STEP {step[0]} {step[1]}
! FIXSTEP
! TRACKDATA
! DUMP
! DEBUG

ANSATZ
QCUCC
QCORDER
QC1 {obj.n_active_spin}
QCPERIODIC''')

def gen_odata(obj, path=Path.cwd()):
    file_path = path / "odata.connect"
    with open(file_path, 'w') as f:
        f.write(
f'''NEWCONNECT 1 3 2.0 20.0 30 0.00025
NEWNEB 25 500 0.00025
NEBK 1.0
NEBMAXERISE 0.01
DIJKSTRA 1
! PERMDIST
EDIFFTOL  1.0D-10
GEOMDIFFTOL  1.0D-10
UPDATES 100
PUSHOPT 0.2 0.001 100
! REOPTIMISEENDPOINTS
BFGSSTEPS 10000
MAXBFGS 0.1
STEPS 1000
BFGSMIN  1.0D-10
! SEARCH 2

NOFRQS

PATH 1000

BFGSTS 1000 5 25 0.001
! BFGSTSPC 0.001
NOHESS
USEDIAG 2

MAXSTEP 0.1
MAXMAX  0.3
TRAD 0.2
MAXERISE 1.0D-2 1.0D0

ANSATZ
QCORDER
QCUCC
QC1 {obj.n_active_spin} {obj.dim}
QCIRCUITMODE 1
DUMPALLPATHS''')

def gen_dinfo(obj, path=Path.cwd()):
    file_path = path / "dinfo"
    with open(file_path, 'w') as f:
        f.write(
f'''AUTORANGE
DELTA 0.01
! FIRST -2.2
! LEVELS 65
MINIMA min.data
TS ts.data

! OPTIONAL KEYWORDS
SCALEBAR 1
! MAXTSENERGY -168.5
NCONNMIN 0
CENTREGMIN
! CONNECTMIN 2
! IDMIN 2
! IDMIN 8
ORDER_BY_ENERGY

! IDENTIFY
! LABELSIZE 5''')

def gen_path(obj, path=Path.cwd()):
    file_path = path / "pathdata"
    with open(file_path, 'w') as f:
        f.write(
f'''! CPUS 4
! CYCLES 100000
! SLURM
! RUNLOCAL
! EVCUT 1.0D-10
NOFRQS

COPYFILES qc_ucc_hamiltonian.dat qc_ucc_index.dat qc_ucc_initial.dat qc_ucc.dat
COPYFILES qc_ucc_order.dat
EXEC $OPTIM_PATH
SYSTEM         QC
NATOMS         {obj.dim}
SEED           1
PERTURB        2.5
DIRECTION      AB
TEMPERATURE    0.1
! DEBUG
EDIFFTOL  1.0D-10
GEOMDIFFTOL  1.0D-10
ITOL           10.1D0

QCUCC
QCORDER
QC1            {obj.n_active_spin} {obj.dim}
CHECKPATHINFO

ADDMIN min.data.info.test
! CONNECTREGION 1 2
! EXTRACTMINFILE''')


def setup(obj, path=Path.cwd(), temp=1, tightconv=1e-7, 
          sloppyconv=1e-6, save=10000, ediff=1e-6, 
          updates=10, maxerise=1e-6, maxit=100000, 
          steps=10000, step=(1.0,0.5), tfac=1.0):
    gen_hamiltonian(obj, path)
    gen_coords(obj, path)
    gen_order(obj, path)
    gen_excitations(obj, path)
    gen_initial(obj, path)
    gen_data(obj, path, temp=temp, tightconv=tightconv, 
             sloppyconv=sloppyconv, save=save, ediff=ediff, 
             updates=updates, maxerise=maxerise, maxit=maxit, 
             steps=steps, step=step, tfac=tfac)
    gen_odata(obj, path)
    gen_dinfo(obj, path)
    gen_path(obj, path)
# def gmin_run(obj, orb, path="./"):
#     file_path = path + "run_GMIN.sh"
#     with open(file_path, 'w') as f:
#         f.write(
# f'''#!/bin/bash
# # Partition to use - generally not needed
# #SBATCH -p MAIN
# #SBATCH --time=2-0:0:0
# #SBATCH -n 1
# #SBATCH -J g-{orb}-{obj.layers}
# #SBATCH --requeue 

# echo $SLURM_NTASKS > nodes.info
# srun hostname >> nodes.info
# echo $USER >> nodes.info
# pwd >> nodes.info

# export OPTIM_PATH=/sharedscratch/gly21/git/softwarewales/OPTIM/builds/ansatz-gfortran/ANSATZOPTIM
# /sharedscratch/gly21/git/softwarewales/GMIN/builds/ansatz-gfortran/ANSATZGMIN > output_GMIN''')

# def optim_run(obj, orb, path="./"):
#     file_path = path + "run_OPTIM.sh"
#     with open(file_path, 'w') as f:
#         f.write(
# f'''#!/bin/bash
# # Partition to use - generally not needed
# #SBATCH -p MAIN
# #SBATCH --time=2-0:0:0
# #SBATCH -n 1
# #SBATCH -J o-{orb}-{obj.layers}
# #SBATCH --requeue 

# echo $SLURM_NTASKS > nodes.info
# srun hostname >> nodes.info
# echo $USER >> nodes.info
# pwd >> nodes.info

# export OPTIM_PATH=/sharedscratch/gly21/git/softwarewales/OPTIM/builds/ansatz-gfortran/ANSATZOPTIM
# /sharedscratch/gly21/git/softwarewales/GMIN/builds/ansatz-gfortran/ANSATZGMIN > output_OPTIM''')

# def path_run(obj, orb, path="./"):
#     file_path = path + "run_PATH.sh"
#     with open(file_path, 'w') as f:
#         f.write(
# f'''#!/bin/bash
# # Partition to use - generally not needed
# #SBATCH -p MAIN
# #SBATCH --time=2-0:0:0
# #SBATCH -n 3
# #SBATCH -J p-{orb}-{obj.layers}
# #SBATCH --requeue 

# echo $SLURM_NTASKS > nodes.info
# srun hostname >> nodes.info
# echo $USER >> nodes.info
# pwd >> nodes.info

# export OPTIM_PATH=/sharedscratch/gly21/git/softwarewales/OPTIM/builds/ansatz-gfortran/ANSATZOPTIM
# /sharedscratch/gly21/git/softwarewales/PATHSAMPLE/builds/gfortran/PATHSAMPLE > output_PATH''')

# def path_init(obj, path="./"):
#     file_path = path + "init_path.sh"
#     with open(file_path, 'w') as f:
#         f.write(
# f'''#!/bin/bash

# cp lowest min.data.info.test
# sed -r -i 's/first.+/ 1.0000000000 1 1.0000000000 1.0000000000 1.0000000000/g' min.data.info.test
# sed -r -i '/\s+[[:digit:]]+$/d' min.data.info.test
# sed -r -i 's/.*[=]\s+//g' min.data.info.test

# /sharedscratch/gly21/git/softwarewales/PATHSAMPLE/builds/gfortran/PATHSAMPLE

# echo -e "1\n1" > min.A
# echo -e "1\n2" > min.B

# sed -i 's/! CYCLES/CYCLES/' pathdata
# sed -i 's/! SLURM/SLURM/' pathdata
# sed -i 's/! RUNLOCAL/RUNLOCAL/' pathdata
# sed -i 's/ADDMIN/! ADDMIN/' pathdata
# sed -i 's/! CONNECTREGION/CONNECTREGION/' pathdata''')

# def merge_sh(obj, path="./"):
#     file_path = path + "merge_min.sh"
#     with open(file_path, 'w') as f:
#         f.write(
# '''#!/bin/bash

# mkdir ./mergedb

# cp ./qc_ucc* ./mergedb/
# cp ./dinfo ./mergedb/
# cp ./odata.connect ./mergedb/
# cp ./pathdata ./mergedb/
# cp ./run_PATH.sh ./mergedb/

# sed -r -i 's/CONNECTREGION/! CONNECTREGION/' ./mergedb/pathdata
# sed -r -i 's/CYCLES/! CYCLES/' ./mergedb/pathdata
# sed -i '1s/^/MERGEDB ..\\n/' ./mergedb/pathdata
# sed -r -i 's/EDIFFTOL\s+1.0D-10/EDIFFTOL  1.0D-5/' ./mergedb/pathdata
# sed -r -i 's/GEOMDIFFTOL\s+1.0D-10/GEOMDIFFTOL  1.0D10/' ./mergedb/pathdata''')


