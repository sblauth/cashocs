CASHOCS
=========

CASHOCS is a computational adjoint-based shape optimization and optimal control software for python.

Installation
------------

- First, install [FEniCS](https://fenicsproject.org/download/), version 2019.1. Note, that
FEniCS should be compiled with PETSc and petsc4py.

- 

- You might also want to install [GMSH](https://gmsh.info/). CASHOCS does not necessarily need this to function properly, but it is required for the remeshing functionality.

- Finally, you can install CASHOCS with the command

        pip3 install cashocs

 You can install the newest (development) version of cashocs with

        pip3 install git+https://github.com/plugged/cashocs

Documentation
-------------

The documentation of the project can be found in the docs folder, just open "index.html"
to view them in a web browser. Moreover, there is also a link to a website containing
the documentation of the demos.


Testing
-------

To run the unit tests for this CASHOCS, run

    pytest

from the ./tests directory.


License
-------

CASHOCS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CASHCOS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with CASHOCS. If not, see <http://www.gnu.org/licenses/>.


Contact / About
---------------

I'm Sebastian Blauth, a PhD student at Fraunhofer ITWM and TU Kaiserslautern,
and I developed this project as part of my work. If you have any questions /
suggestions / feedback, etc., you can contact me via
[sebastian.blauth@itwm.fraunhofer.de](mailto:sebastian.blauth@itwm.fraunhofer.de).
