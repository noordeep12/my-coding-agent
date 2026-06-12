my-coding-agent
===============

A minimal Python agent framework for running LLM-powered coding agents against
local LLM servers. The public API re-exports the agent loop, the LLM client, the
decorator-based tool registry, the context-handoff state object, and the
exception hierarchy.

.. toctree::
   :maxdepth: 2

   api


Quickstart
----------

.. code-block:: python

   from my_coding_agent import Agent, tool

   @tool
   def add(a: int, b: int) -> int:
       """Add two integers.

       Args:
           a: First addend.
           b: Second addend.

       Returns:
           The sum of ``a`` and ``b``.
       """
       return a + b


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
