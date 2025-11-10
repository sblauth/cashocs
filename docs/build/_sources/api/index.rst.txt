cashocs API Reference
=====================

Below you find the documented public API of cashocs.

.. autosummary::
   :toctree: generated
   :recursive:

   cashocs

.. note::

   Below, you will find only the public API of cashocs. However, cashocs also has a
   private API. The way python is structured does not allow to make attributes /
   modules, etc. private. We use the definition that every function, class, or module 
   starting with a leading underscore ``_`` is private. The contents of every private
   module are also assumed to be private, as are, e.g., all attributes and methods of 
   a private class. Additionally, every object starting with two leading underscores
   ``__`` is also assumed to private and uses python's name mangling to protect 
   unintended access.

.. warning::

   Users should not use private objects, functions, methods, classes, or modules. 
   Doing so can lead to unintended behavior, errors, and can break the code.
   Moreover, we are trying to make cashocs public API stable and document all changes 
   made to the public API. There is no warranty that the private API stays the same
   and changes to it are not documented. 

.. note::

   However, cashocs private API is also documented as a help for users to understand 
   what is going on beneath the surface and for developers to allow for easier 
   contribution. You can find the private API in cashocs' source code, which can be 
   easily found and viewed by most IDEs nowadays.


