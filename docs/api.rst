API Reference
=============

The package ``my_coding_agent`` re-exports the core types listed in its
``__all__``. Each object is documented once, in the submodule that defines it.

.. automodule:: my_coding_agent
   :no-members:


Engine
------

.. automodule:: my_coding_agent.engine.agent
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.llm
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.llm.errors
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.llm.parsing
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.checkpoint
   :members:
   :show-inheritance:


Tool registry
-------------

.. automodule:: my_coding_agent.engine.tool_registry.registry
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.tool_registry.converter
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.tool_registry.skills
   :members:
   :show-inheritance:


Tool execution
--------------

.. automodule:: my_coding_agent.engine.tool_execution
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.tool_execution.schema
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.engine.tool_execution.concurrency
   :members:
   :show-inheritance:


Tool routing
------------

.. automodule:: my_coding_agent.engine.routing
   :members:
   :show-inheritance:


Pipeline
--------

.. automodule:: my_coding_agent.pipeline.dag
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.pipeline.context
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.pipeline.node
   :members:
   :show-inheritance:


.. automodule:: my_coding_agent.pipeline.schema
   :members:
   :show-inheritance:


Pipeline nodes
--------------

.. automodule:: my_coding_agent.pipeline.nodes.context_guard
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.pipeline.nodes.tool_routing
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.pipeline.nodes.tool_dispatch
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.pipeline.nodes.llm_call
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.pipeline.nodes.finalize_step
   :members:
   :show-inheritance:


Observability
-------------

.. automodule:: my_coding_agent.observability.recorder
   :members:
   :show-inheritance:


Utils
-----

.. automodule:: my_coding_agent.utils.logging_core
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.utils.terminal_ui
   :members:
   :show-inheritance:

.. automodule:: my_coding_agent.utils.exceptions
   :members:
   :show-inheritance:
