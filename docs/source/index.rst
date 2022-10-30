.. cashocs documentation master file, created by
   sphinx-quickstart on Fri Sep  4 07:56:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cashocs Documentation
=====================

cashocs is a software for solving PDE constrained optimization problems. It is a python package which is based on the finite element software FEniCS and enables the fast and efficient solution of optimization problems constrained by partial differential equations in the fields of shape optimization and optimal control.

.. toctree::
   :maxdepth: 1
   :hidden:
   
   About <about/index>
   User Guide <user/index>
   API Reference <api/index>
   Development <development/index>
   release_notes

**Version**: |release|

**Useful links**:
:ref:`Installation <installation_instructions>` |
`Source Repository <https://github.com/sblauth/cashocs>`_ |
`Issues & Ideas <https://github.com/sblauth/cashocs/issues>`_ |
:ref:`Tutorial <tutorial_index>` |
:ref:`Config File Documentation <config_shape_optimization>`

.. grid:: 2
   :gutter: 3 3 3 3

   .. grid-item-card:: Getting Started
      :img-top: icons/getting_started.svg

      Are you new to cashocs or want to find information about its installation
      and other parts? Then go no further and start here.
      
      +++ 

      .. button-ref:: about/nutshell
         :expand:
         :color: secondary
         :click-parent:

         cashocs in a nutshell

   .. grid-item-card:: User Guide
      :img-top: icons/user_guide.svg

      The user guide provides a tutorial for cashocs which explains each of its 
      features and the mathematical background detailedly.

      +++

      .. button-ref:: user/index
         :expand:
         :color: secondary
         :click-parent:

         To the user guide

   .. grid-item-card:: API Reference
      :img-top: icons/api.svg

      The reference guide contains a detailed explanation of cashocs public API: Each
      class, function, and module you can use and its parameters are described in
      detail here.

      +++

      .. button-ref:: api/index
         :expand:
         :color: secondary
         :click-parent:

         To the reference guide

   .. grid-item-card:: Contributor's Guide
      :img-top: icons/development.svg

      If you want to contribute to cashocs, best start here.

      +++

      .. button-ref:: development/index
         :expand:
         :color: secondary
         :click-parent:

         To the contributor's guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
