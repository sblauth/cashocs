Topology Optimization Problems
==============================

In this part of the tutorial, we investigate how topology optimization problems can
be treated with cashocs.

.. toctree::
   :maxdepth: 1
   :caption: List of all topology optimization demos:

   demo_poisson_clover.md
   demo_cantilever.md
   demo_pipe_bend.md


.. note::

    As topology optimization problems are very involved from a theoretical point of view, it is, at the moment, not possible to automatically derive topological derivatives. Therefore, cashocs cannot be used as "black-box" solver for topology optimization problems in general.

    Moreover, our framework for topology optimization of using a level-set function is quite flexible, but requires a lot of theoretical understanding. In :ref:`demo_poisson_clover`, we briefly go over some theoretical foundations required for using cashocs' topology optimization features. We refer the reader, e.g., to `Sokolowski and Novotny - Topological Derivatives in Shape Optimization <https://doi.org/10.1007/978-3-642-35245-4>`_ for an exhaustive treatment of these topics.
