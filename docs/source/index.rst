.. cashocs documentation master file, created by
   sphinx-quickstart on Fri Sep  4 07:56:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: cashocs_banner.jpg
  :width: 800
  :alt: cashocs

Documentation of cashocs
========================

cashocs is a finite element software for the automated solution of shape optimization and optimal control problems. It is used to solve problems in fluid dynamics and multiphysics contexts. Its name is an acronym for computational adjoint-based shape optimization and optimal control software and the software is written in Python.

.. toctree::
   :maxdepth: 1
   :hidden:
   
   About <about/index>
   User Guide <user/index>
   API Reference <api/index>
   CLI Reference <cli/index>
   Development <development/index>
   release_notes

**Version**: |release|

**Useful links**:
:ref:`Installation <installation_instructions>` |
`Source Code <https://github.com/sblauth/cashocs>`_ |
:ref:`Tutorial <tutorial_index>` |
`Applications <https://www.itwm.fraunhofer.de/en/departments/tv/products-and-services/shape-optimization-cashocs-software.html>`_ |
:ref:`Config File Documentation <config_shape_optimization>`

.. grid:: 2
   :gutter: 3 3 3 3

   .. grid-item-card:: Getting Started
      :img-top: icons/getting_started.svg
      
      Here, you can find information on how to get started with cashocs.
      
      +++ 

      .. button-ref:: about/nutshell
         :expand:
         :color: secondary
         :click-parent:

         cashocs in a nutshell

   .. grid-item-card:: User Guide
      :img-top: icons/user_guide.svg

      The user guide explains cashocs' features and background detailedly.

      +++

      .. button-ref:: user/index
         :expand:
         :color: secondary
         :click-parent:

         To the user guide

   .. grid-item-card:: API Reference
      :img-top: icons/api.svg
      
      The reference guide documents the public API of cashocs.

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
