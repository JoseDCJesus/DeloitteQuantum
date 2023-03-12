from qiskit.algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit_aer.primitives import Estimator
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import TwoQubitReduction
from qiskit_nature.drivers.second_quantization import ElectronicStructureMoleculeDriver, ElectronicStructureDriverType
from qiskit_nature.drivers import Molecule
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
import numpy as np
import pennylane as qml
import time

#Define Molecule

#Get Structure from xyz file
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



#Define VQE

#Define and run driver
driver = ElectronicStructureMoleculeDriver(
    molecule=molecule,
    basis="cc-pVDZ",
    driver_type=ElectronicStructureDriverType.PYSCF)

q_molecule = driver.run()


#Convert Fermionic operators to qubit operatos and get Electronic Structure Problem
qubit_converter = QubitConverter(ParityMapper(), two_qubit_reduction=True)
es_problem = ElectronicStructureProblem(driver, [ActiveSpaceTransformer(6,6)])

#Get Hamiltonian/Qubit Operator
second_q_op = es_problem.second_q_ops()
qubit_op = qubit_converter.convert(second_q_op['ElectronicEnergy'])
reducer = TwoQubitReduction(es_problem.num_particles)
qubit_op = reducer.convert(qubit_op)
num_qubits = es_problem.num_spin_orbitals-2

#Define Initial Ansatz = HF + UCCSD

HF_state = HartreeFock(num_spin_orbitals=es_problem.num_spin_orbitals-2,
                                                       num_particles=es_problem.num_particles,
                                                       qubit_converter=qubit_converter)

ansatz = UCCSD(qubit_converter=qubit_converter,
                     num_particles=es_problem.num_particles,
                     num_spin_orbitals=es_problem.num_spin_orbitals-2,
                     initial_state=HF_state)

VQE_instance = VQE(ansatz=ansatz,
                   optimizer=COBYLA(),
                   estimator=Estimator(approximation=False),
                   initial_point=np.zeros(ansatz.num_parameters),
                   callback=None)

#Run ADAPT-VQE
AdaptVQE_instance = AdaptVQE(VQE_instance)

result = AdaptVQE_instance.compute_minimum_eigenvalue(qubit_op)
print(result)
print(result.optimal_circuit.decompose())
AdaptVQE_instance.ansatz = result.optimal_circuit
print(AdaptVQE_instance.ansatz.decompose().decompose())

f= open('Ansatz.txt', 'w')
f.write(str( result.optimal_circuit.decompose().draw('text')))
f.write('\n')
f.write(str( result.optimal_circuit.decompose().decompose().draw('text')))
f.close()