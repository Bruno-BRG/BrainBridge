# Task Management & Coding Guide for Copilot

*Extends your **Coding Standards and Best Practices***

---

## 1 · Purpose

| Goal                       | Why it matters                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Single source of truth** | `task.md` tracks every feature, bug‑fix, refactor or chore.                                              |
| **Automated visibility**   | Copilot must *always* keep `task.md` in sync with the codebase so you (Momobubu) see real‑time progress. |

---

## 2 · `task.md` File Structure

```md
# Project Tasks

## Backlog
- [ ] short‑imperative‑title – _one‑line description_

## In Progress
- [ ] short‑imperative‑title – _one‑line description_

## Review / Testing
- [ ] …

## Done ✅
- [x] YYYY‑MM‑DD implemented‑feature‑xyz – passed unit & integration tests
```

**Hard Rules**

1. One task per line.
2. Keep the four sections in the exact order above.
3. No nested check‑boxes – flat lists only.
4. Append *new* tasks to the **Backlog** bottom.

---

## 3 · Task Lifecycle

| Stage                              | Trigger                                      | Copilot Action                                                                 |
| ---------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------ |
| **Backlog → In Progress**          | Copilot begins implementing the task         | Move the line to **In Progress**                                               |
| **In Progress → Review / Testing** | Code compiles & unit tests are green locally | Move to **Review / Testing** and append the commit hash                        |
| **Review / Testing → Done**        | Project test‑suite *and* manual checks pass  | Change `[ ]` → `[x]`, prepend today’s date `YYYY‑MM‑DD`, then move to **Done** |
| **Any → Backlog**                  | Task blocked or postponed                    | Move back to **Backlog** and add `« BLOCKED: reason »`                         |

---

## 4 · Markdown Checkbox Syntax

| State     | Markdown     |
| --------- | ------------ |
| Unchecked | `- [ ] task` |
| Checked   | `- [x] task` |

*Never* invent alternative symbols.

---

## 5 · Commit Message Convention

```text
<type>(#<task‑line‑number>): <summary>

BODY: detailed explanation of change, limitations, and testing notes.
```

`<type>` ∈ { **feat**, **fix**, **refactor**, **docs**, **chore**, **test** }.

---

## 6 · Automatic Summary Note (Copilot)

At the **end of every Copilot‑generated message** append:

```md
### Task Update
- Moved <task‑title> → <new‑section>
- Added commit <hash>
```

This fulfils your requirement *“summary of the changes you made in the code”*.

---

## 7 · Example Workflow

1. **Backlog** contains `- [ ] add‑login‑endpoint` (line 12).
2. Copilot starts coding → moves line 12 to **In Progress**.
3. Tests pass → moves the task to **Review / Testing** and adds the commit hash.
4. You approve → Copilot moves it to **Done** with `[x] 2025‑05‑28 …`.

---

## 8 · Quality Gates

* No task may skip a stage.
* A PR **cannot be merged** while **In Progress** or **Review / Testing** lists are non‑empty.
* Linting, unit tests, and integration tests **must** be green before a task moves to **Done**.

---

## 9 · House‑Keeping

* Archive tasks older than 90 days in `archive/task‑YYYY‑MM.md`.
* **Never** delete history.

---

## 10 · Coding Style & Documentation Standards — *“NASA‑grade OOP”*

1. **Utmost Object‑Oriented Design**

   * Every class lives in its *own* file (one‑class‑per‑file rule).
   * Encapsulate behaviour; expose minimal, purposeful public APIs.
   * Prefer composition over inheritance unless a clear *is‑a* relationship exists.

2. **File‑Level Header Comment** (top ≈ 10 lines)

   ```python
   """
   Class:   OrbitalTransferCalculator
   Purpose: Calculates optimal Δv for Hohmann and bi‑elliptic transfers.
   Author:  Copilot (NASA‑style guidelines)
   Created: 2025‑05‑28
   Notes:   Follows Task Management & Coding Guide for Copilot v2.0
   """
   ```

3. **Function / Method Docstrings**

   * One‑sentence summary on the first line.
   * `Args:` list each parameter, type, units (if applicable), and expected range.
   * `Returns:` exact return type and description.
   * `Raises:` enumerates exceptions with reasons.
   * Example:

     ```python
     def delta_v(self, mu: float, r1: float, r2: float) -> float:
         """
         Compute total Δv for a Hohmann transfer between two circular orbits.

         Args:
             mu (float): Standard gravitational parameter [km³/s²].
             r1 (float): Radius of the initial orbit [km] (r1 > 0).
             r2 (float): Radius of the target orbit [km] (r2 > 0).

         Returns:
             float: Total Δv required [km/s].

         Raises:
             ValueError: If r1 or r2 are non‑positive.
         """
     ```

4. **Implementation Rigor**

   * Follow SOLID principles, Clean Code, and MISRA‑C‑like safety checks (adapted for Python).
   * Write *defensive* code: validate inputs, use explicit exceptions, avoid magic numbers.
   * Opt for immutability where possible; document side‑effects clearly.

5. **Testing & Verification**

   * Provide exhaustive unit tests per class file (`tests/test_<class>.py`).
   * Tests cover nominal cases, edge cases, and failure modes.
   * Aim for ≥ 95 % branch coverage; no unchecked exceptions.

6. **Continuous Documentation**

   * Keep README and API reference up‑to‑date with any public‑facing changes.
   * Auto‑generate docs (e.g., Sphinx) as part of CI.

---

### EOF
