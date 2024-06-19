# xpfem
A python package for finite element method (FEM) in solid mechanics

You should download all the code and install the following packages:
* numpy
* scipy
* pyvista

## Features:
* linear elasticity
* 2D Quadrilateral element (Q1,Q2)
* 3D Hexahedral element (Q1,Q2)
* BBar element for 3D Q1 element
* Perfect post-processing
* Export VTK file

## Run code
You can install the packages by
1. pip install numpy
2. pip install scipy
3. pip install pyvista

Run the test.py or other test file, you can get the displace and stress as following

<div align="center">
    <img src="https://github.com/Qinxiaoye/xpfem/blob/main/figure/beam.png">
</div>

<div align="center">
    <img src="https://github.com/Qinxiaoye/xpfem/blob/main/figure/pipe.png">
</div>

## Preparing input files from ANSYS
See <https://github.com/Qinxiaoye/FEM2D> to learn how to creat the input files from ANSYS
