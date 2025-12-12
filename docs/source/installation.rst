Installation
============
Grew-TSE relies on the great work carried out on `Grew (Graph Rewriting for NLP) <https://grew.fr/>`_ and therefore a number of installations must be carried out as a prerequisite. Without these additional installation steps you will not be able to use this package.

Opam & Grewpy
-------------
You must first install `Opam <https://opam.ocaml.org/>`_. This will allow the `grewpy` package (automatically installed with Grew-TSE) to work properly.
If using Windows, please note that we have only tested this when using WSL.
You can install Opam on Linux using the following command:

.. code-block:: bash

   bash -c "sh <(curl -fsSL https://opam.ocaml.org/install.sh)"

Or on Windows Powershell:

.. code-block:: powershell

   Invoke-Expression "& { $(Invoke-RestMethod https://opam.ocaml.org/install.ps1) }"

To initiate the opam setup, you can then run:

.. code-block:: bash

   opam init

These scripts automatically detect your system architecture and carry out the installation.
More details on installing Opam are provided `here <https://opam.ocaml.org/doc/Install.html>`_.

The necessary grewpy backend can then be installed with the following commands:

.. code-block:: bash

   opam remote add grew "https://opam.grew.fr"
   opam update
   opam install grewpy_backend

You will need to tell your system where to find OPAM's bin directory. Run the following to have this permanently added:

.. code-block:: bash

   echo 'eval $(opam env)' >> ~/.bashrc

Grew-TSE Pipeline
--------
You can install Grew-TSE with the below command using pip, which should automatically install any dependencies required.

.. code-block:: bash

    pip install grew-tse

The main pipeline in Grew-TSE requires Python 3.8 or higher and the below Python packages.
- ``conllu==6.0.0``
- ``grewpy==0.6.0``
- ``numpy==2.2.5``
- ``pandas==2.2.3``.
These should be installed automatically using the above install of Grew-TSE.

Evaluation
----------
If you want to use any of the Grew-TSE evaluation tools, then you'll also need the `eval` dependencies. Install them like so:

.. code-block:: bash

   pip install grew-tse[eval]

For any issues with ```grewpy```, visit the `Grew documentation <https://grew.fr/usage/python/>`_ for more information on the installation as well as how to test whether your installation was successful.
