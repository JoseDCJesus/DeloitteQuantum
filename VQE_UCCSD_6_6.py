import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import ParityMapper, QubitConverter
from qiskit.opflow import TwoQubitReduction
from qiskit.algorithms import MinimumEigensolver, VQEResult
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.primitives import Estimator
import time
from qiskit.algorithms.optimizers import COBYLA

# Loading your IBM Quantum account(s)
IBMQ.save_account('7d51f5c5e5c7521015c58312ad6428df9fc7dcf957d3159dfba1f25c77afdbee41b8ae64d75b87285e51cd8d0b28c172a4a13c2491283414a1d489281fb5a2f1')
provider = IBMQ.load_account()

#Import pennylane
import pennylane as qml

#Get Structure
symbols, coordinates = qml.qchem.read_structure('CO2/step_000.xyz')

laux = []

for i in range(4):
    aux = []
    for j in range(3):
        aux.append(float(coordinates[i*3+j])*0.52917721067121)
    laux.append(aux)
    

from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)

# Define Molecule

molecule = Molecule(
    # Coordinates in Angstrom
    geometry=[
        ["Cu", laux[0]],
        ["O", laux[1]],
        ["C", laux[2]],
        ["O", laux[3]]
    ],
    multiplicity=1,  # = 2*spin + 1
    charge=1,
)

driver = ElectronicStructureMoleculeDriver(
    molecule=molecule,
    basis="cc-pVDZ",
    driver_type=ElectronicStructureDriverType.PYSCF)

#Define Active Space with 6 electrons and 6 orbitals
problem = ElectronicStructureProblem(
    driver,
    [ActiveSpaceTransformer(6,6)])

second_q_ops = problem.second_q_ops()  # Get 2nd Quant OP
num_spin_orbitals = problem.num_spin_orbitals
num_particles = problem.num_particles

print(num_particles)
print(num_spin_orbitals)

#Convert Fermionic Operators to Qubit Operators
mapper = ParityMapper()
converter = QubitConverter(mapper, two_qubit_reduction=True)
hamiltonian = second_q_ops['ElectronicEnergy']
qubit_op = converter.convert(hamiltonian)
reducer = TwoQubitReduction(num_particles)
qubit_op = reducer.convert(qubit_op)
num_particles = num_particles
num_spin_orbitals = num_spin_orbitals -2

# Define Ansatz
Ansatz = HartreeFock(num_spin_orbitals//2, num_particles, converter)

Ansatz.append( UCCSD(num_spin_orbitals//2, num_particles, converter), [i for i in range(num_spin_orbitals)])

print(Ansatz.draw())
print(Ansatz.decompose().decompose().decompose().decompose().count_ops())
      
#Noiseless Simulation - 10000 shots

from qiskit.algorithms import MinimumEigensolver, VQEResult

# Define a custome VQE class to orchestra the ansatz, classical optimizers, 
# initial point, callback, and final result
class CustomVQE(MinimumEigensolver):
    
    def __init__(self, estimator, circuit, optimizer, callback=None):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        
    def compute_minimum_eigenvalue(self, operators, aux_operators=None):
                
        # Define objective function to classically minimize over
        def objective(x):
            # Execute job with estimator primitive
            job = self._estimator.run([self._circuit], [operators], [x], shots=10000)
            # Get results from jobs
            est_result = job.result()
            # Get the measured energy value
            value = est_result.values[0]
            # Save result information using callback function
            if self._callback is not None:
                self._callback(value)
            return value
            
        # Select an initial point for the ansatzs' parameters
        x0 = np.pi/4 * np.random.rand(self._circuit.num_parameters)
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        
        # Populate VQE result
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenvalue = res.fun
        result.optimal_parameters = res.x
        return result
    
#Define Optimizer
optimizer = COBYLA(maxiter=400)

symbols, coordinates = qml.qchem.read_structure('CO2/step_000.xyz')

laux = []

for i in range(4):
    aux = []
    for j in range(3):
        aux.append(float(coordinates[i*3+j]))
    laux.append(aux)
    
ref = np.sqrt((laux[0][0]-laux[1][0])**2 + (laux[0][1]-laux[1][1])**2 + (laux[0][2]-laux[1][2])**2)

print(ref)

def Energ_calc(jj):
    
    if it < 10:
        straux = 'CO2/step_00'+str(it)+'.xyz'
    else:
        straux = 'CO2/step_0'+str(it)+'.xyz'
    
    print(straux)
    
    laux = []

    for i in range(4):
        aux = []
        for j in range(3):
            aux.append(float(coordinates[i*3+j])*0.52917721067121)
        laux.append(aux)

    molecule = Molecule(
    # Coordinates in Angstrom
    geometry=[
        ["Cu", laux[0]],
        ["O", laux[1]],
        ["C", laux[2]],
        ["O", laux[3]]
    ],
    multiplicity=1,  # = 2*spin + 1
    charge=1,
    )

    
    driver = ElectronicStructureMoleculeDriver(
    molecule=molecule,
    basis="cc-pVDZ",
    driver_type=ElectronicStructureDriverType.PYSCF)

    problem = ElectronicStructureProblem(
    driver,
    [ActiveSpaceTransformer(6,6)])

    second_q_ops = problem.second_q_ops()  # Get 2nd Quant OP
    num_spin_orbitals = problem.num_spin_orbitals
    num_particles = problem.num_particles
    
    mapper = ParityMapper()
    converter = QubitConverter(mapper, two_qubit_reduction=True)
    hamiltonian = second_q_ops['ElectronicEnergy']
    qubit_op = converter.convert(hamiltonian)
    reducer = TwoQubitReduction(num_particles)
    qubit_op = reducer.convert(qubit_op)
    num_particles = num_particles
    num_spin_orbitals = num_spin_orbitals -2
    
    ansatz_cp = Ansatz.copy()
     
    # Define instance of qiskit-terra's Estimator primitive
    estimator = Estimator([ansatz_cp], [qubit_op])
    # Setup VQE algorithm

    # Define a simple callback function
    intermediate_info = []
    def callback(value):
            intermediate_info.append(value)

    custom_vqe = CustomVQE(estimator, ansatz_cp, optimizer, callback=callback)

    # Run the custom VQE function and monitor execution time
    
    res_sim = custom_vqe.compute_minimum_eigenvalue(qubit_op)
    
    
    cmp_gs = problem.interpret(res_sim).total_energies[0].real
    
    return cmp_gs *  27.211


ener_val = []
dist_val = []

#Run Algorithm for every step
start = time.time()
for it in range(24):
    
    if it < 10:
        straux = 'CO2/step_00'+str(it)+'.xyz'
    else:
        straux = 'CO2/step_0'+str(it)+'.xyz'
    symbols, coordinates = qml.qchem.read_structure(straux)

    laux = []

    for i in range(3):
        aux = []
        for j in range(3):
            aux.append(float(coordinates[i*3+j]))
        laux.append(aux)


    dist_val.append(np.sqrt((laux[0][0]-laux[1][0])**2 + (laux[0][1]-laux[1][1])**2 + (laux[0][2]-laux[1][2])**2)-ref)
    print('AQUI')
    print(dist_val[-1])
        
    ener_val.append(Energ_calc(it))
    print(ener_val[-1])
    
end = time.time()
print(f'execution time (s): {end - start:.2f}')

f = open('UCCSD(6,6)_results.txt', 'w')
f.write(str(dist_val))
f.write('\n')
f.write(str(ener_val))
f.close()