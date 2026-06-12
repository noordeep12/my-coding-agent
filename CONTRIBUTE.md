# Contributing Guidelines

# Mindset
## 1. Core Engineering Mindset
- Think as a **creator**, not a user.
- Always start with **WHY**: why does this system exist, and what problem does it solve?
- Then understand **HOW** it works internally and across systems.
- Be able to explain: problem → effort → solution clearly.
- Focus on design decisions; this is where engineering value exists (AI can implement, but context-driven design is human responsibility).

---

## 2. Problem Understanding (Before Coding)
Before implementation:
- What exact problem are we solving?
- Why does this problem exist?
- Who is affected?
- What outcome defines success?
- What assumptions are we making?

Do not start implementation without clarity on problem + assumptions.

---

## 3. Scope Control
- Keep changes small and focused.
- Avoid unrelated refactoring.
- Prefer incremental improvements over large rewrites.
- If scope grows, split into multiple tasks.

---

## 4. Complexity Control
- Minimize cognitive load in code.
- If code is hard to understand quickly, it is too complex.
- Prefer simple, explicit logic over clever implementations.
- Avoid unnecessary abstractions and hidden behavior.

---

## 5. Modularity & Information Hiding
- Design systems as independent modules.
- Each module must have a single responsibility.
- Hide internal complexity behind simple interfaces.
- Good modules do not require reading internal code to understand usage.

---

## 6. Change Isolation
- Prefer changes contained in a single file.
- Multi-file changes must be justified.
- Reduce cross-module coupling.
- Avoid widespread changes for small features.

---

## 7. Dependencies
- Minimize dependencies between components.
- Avoid circular dependencies completely.
- Prefer shallow dependency graphs.

---

## 8. Abstraction Rules
- Abstraction exists to reduce complexity, not increase it.
- Create abstractions only when multiple real use cases exist.
- Avoid pass-through methods.
- Generic components must solve real repeated problems.

---

## 9. Defaults & Edge Cases
- Design APIs so common cases are simple by default.
- Handle common edge cases internally where appropriate.
- Reduce special-case handling at call sites.

---

## 10. Data & State Management
- Keep state local when possible.
- Minimize variable usage scope.
- Avoid spreading state across many components.

---

## 11. Error Handling & Fail-Fast
- Fail-fast: detect and fail early when something is invalid.
- Do not ignore errors silently.
- Handle errors close to source when possible.
- Propagate only when caller can meaningfully act.
- Avoid unnecessary complexity in error chains.

---

## 12. Reliability Thinking
- Systems must work correctly even when things go wrong.
- Expect hardware failures, software bugs, and human mistakes.
- Design for fault tolerance and recovery.
- Test system behavior under failure conditions.

---

## 13. Scalability Thinking
When designing scalable systems:
- Define load parameters (requests/sec, reads/writes, concurrency).
- Identify performance bottlenecks (p50, p95, p99 latency).
- Choose between scaling up vs scaling out.
- Understand trade-offs: scalability increases complexity.

---

## 14. Maintainability
- Reduce long-term complexity.
- Favor designs that are easy to modify and debug.
- Hide backend complexity behind clean interfaces.
- Optimize for engineer productivity and system health.

---

## 15. Systems Thinking (Why / How)
- Always ask WHY a system is designed this way.
- Understand HOW systems work internally (e.g. network, storage, APIs).
- Know common system design patterns.
- Be able to reason about trade-offs and constraints.

---

## 16. Code Readability Rule
- Code must be understandable without jumping across multiple files.
- If understanding requires tracing many layers, the design is too complex.
- Prefer local reasoning over distributed reasoning.

---

## 17. Code Review Standards
All changes are reviewed for:
- Correctness
- Simplicity
- Security
- Edge cases
- Maintainability
- Hidden complexity

---

## 18. Testing Requirements
- Every feature must include tests.
- Bug fixes must include a regression test.
- Cover edge cases, not just happy paths.
- Tests validate behavior, not implementation details.

---

## 19. Observability
Production systems must be observable:
- Logs for debugging
- Metrics for performance and health
- Clear failure signals

If it runs in production, its behavior must be explainable.

---

## 20. Ownership
- You own your code after deployment.
- If it breaks, you are responsible for fixing it.

---

## 21. Security Baseline
- Validate all external inputs.
- Assume all external data is untrusted.
- Never expose secrets in logs or errors.
- Use least privilege access principles.

---

## 22. Study & Learning Principles
- Close material and summarize concepts from memory.
- Explain concepts simply (What / Why / How / Limits).
- Compare similar systems (e.g. replication vs partitioning).
- Study failure cases, not just ideal behavior.
- Read actively (ask why design decisions exist).

---

## 23. System Design Thinking Examples
- APIs exist for interoperability, reuse, and abstraction.
- Reliability = system continues working under faults.
- Scalability = ability to handle increasing load efficiently.
- Maintainability = ability to evolve system without excessive complexity.

Examples:
- Twitter fan-out: solve read-load scaling by precomputing timelines.
- LinkedIn profile as document: avoid expensive joins using document-based access.

---

## 24. Key Principle
Good engineering is about reducing complexity while increasing capability.
Bad engineering is hiding complexity without control.

