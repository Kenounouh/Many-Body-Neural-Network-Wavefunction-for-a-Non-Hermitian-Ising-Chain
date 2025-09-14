# Many-Body-Neural-Network-Wavefunction-for-a-Non-Hermitian-Ising-Chain
This repository provide the codes that support the results in our paper: https://arxiv.org/abs/2506.11222

# Neural Quantum States

Neural-network-based variational wavefunctions or neural quantum states (NQSs), such as restricted Boltzmann machines (RBMs), multilayer perceptrons (MLPs), and recurrent neural networks (RNNs) have demonstrated remarkable success in
approximating ground states (quantum many-body wave function ansätzes) of complex quantum systems with the help of Variational Monte Carlo (VMC). <a href="https://arxiv.org/abs/2506.11222" target="_blank">In our paper</a>, we show that these NQSs can provide accurate estimations of ground-state properties, correlation functions as well as quantum phase transitions in many-body non-Hermitian (NH) systems.

## Content
This repository contains the following folders:
* **ED**: an implementation of exact diagonalization for the the 1D non-Hermitian PT-symmetric Transverse-field Ising Model (TFIM)

* **RNN**: an implementation of the RNN Wave Function for the finding of the ground state of the 1D non-Hermitian PT-symmetric Transverse-field Ising Model (NH TFIM).

* **RBM**: an implementation of the RBM Wave Function for the finding of the ground state of the 1D non-Hermitian PT-symmetric Transverse-field Ising Model (NH TFIM).

* **MLP**:an implementation of the MLP Wave Function for the finding of the ground state of the 1D non-Hermitian PT-symmetric Transverse-field Ising Model (NH TFIM).

* **TL**: an implementation of the RNN, RBM, MLP  Wave Function with transfer learning for the finding of the average magnetization in 1D non-Hermitian PT-symmetric Transverse-field Ising Model (NH TFIM).


To learn more about this approach, you can check out our paper on Physical Review Research: https://arxiv.org/abs/2506.11222

For further questions or inquiries, please feel free to send an email to lavoisier.wahkenounouh@mpl.mpg.de. We are looking forward to future contributions.

## License
The [license](https://github.com/Kenounouh/Many-Body-Neural-Network-Wavefunction-for-a-Non-Hermitian-Ising-Chain/edit/main/LICENSE) of this work is derived from the BSD-3-Clause license. Ethical clauses are added to promote good uses of this code.

## Citing
```bibtex
@article{wah2025many,
  title={Many-Body Neural Network Wavefunction for a Non-Hermitian Ising Chain},
  author={Wah, Lavoisier and Zen, Remmy and Kunst, Flore K},
  journal={arXiv preprint arXiv:2506.11222},
  year={2025}
}
```
## Dependencies
Our implementation works with the packages in the ``requirements.txt'' file

## Run
For simplicity all the NQSs where stored as jupyter notebooks.


