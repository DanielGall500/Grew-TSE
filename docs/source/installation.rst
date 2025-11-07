Installation
============
Grew-TSE relies on the great work carried out on `Grew (Graph Rewriting for NLP) <https://grew.fr/>`_ and therefore a number of installations must be carried out as a prerequisite. Without these additional installation steps you will not be able to use this package.

Opam & Grewpy
-------------
You must first install the official Python package for Grew, *Grewpy*, on your system using `Opam <https://opam.ocaml.org/>`_.
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

As a final prerequisite, you may then install the grewpy package itself using pip:

.. code-block:: bash

   pip install grewpy

You can now test the installation was successful by running the following command.
It should output 'connected to port...'.

.. code-block:: bash

   echo "import grewpy" | python

Visit the `Grew documentation <https://grew.fr/usage/python/>`_ for more information on the installation as well as how to test whether your installation was successful.

Grew-TSE
--------
You can install Grew-TSE with the below command using pip, which should automatically install any dependencies required.

.. code-block:: bash

    pip install grewtse

Grewpy requires Python 3.8 or higher and the below Python packages.
- ``conllu==6.0.0``
- ``grewpy==0.6.0``
- ``numpy==2.2.5``
- ``pandas==2.2.3``
- ``plotnine==0.14.5``
- ``tokenizers==0.21.1``
- ``torch==2.7.0``
- ``transformers==4.52.3``
These should be installed automatically using the above install of Grew-TSE.
