# Task Management Guide for Copilot

This guide extends your existing **Coding Standards and Best Practices** and tells Copilot *exactly* how to drive the development workflow by reading and updating a `task.md` checklist.

---

## 1 · Purpose

* **Single source of truth**: `task.md` records every feature, bug‑fix, refactor or chore.
* **Automated visibility**: Copilot must *always* synchronise code changes with the checklist so you (Momobubu) can see real‑time progress.

---

## 2 · `task.md` file structure

```md
# Project Tasks

## Backlog
- [ ] short‑imperative‑title ­– _one line description_

## In Progress
- [ ] same‑format‑as‑above

## Review / Testing
- [ ] …

## Done ✅
- [x] implemented‑feature‑xyz – passed unit & integration tests
```

**Rules**

1. **One line per task.**
2. Keep sections in the order shown above.
3. Never nest check‑boxes – flat lists only.
4. Place *new* tasks at the bottom of **Backlog**.

---

## 3 · Task lifecycle

| Stage                              | Trigger                                     | Copilot action                                                              |
| ---------------------------------- | ------------------------------------------- | --------------------------------------------------------------------------- |
| **Backlog → In Progress**          | Copilot starts writing code for the task    | Move line to *In Progress*                                                  |
| **In Progress → Review / Testing** | All code compiles & unit tests pass locally | Move line to *Review / Testing* and append commit hash                      |
| **Review / Testing → Done**        | Your test suite & manual checks succeed     | Replace `[ ]` with `[x]`, prepend today’s date YYYY‑MM‑DD, move to **Done** |
| **Any → Backlog**                  | Task blocked or postponed                   | Move line back to *Backlog* and add « BLOCKED: reason » comment             |

---

## 4 · Markdown checkbox syntax

* **Unchecked** : `- [ ] task`
* **Checked**    : `- [x] task`

Copilot must **never** invent alternative symbols.

---

## 5 · Commit message convention

```
<type>(#<task‑line‑number>): <summary>

BODY: detailed explanation of change, limitations, and testing notes.
```

`<type>` ∈ {feat, fix, refactor, docs, chore, test}.

---

## 6 · Automatic summary note

At the **end of every Copilot‑generated message** it must append:

```
### Task Update
- Moved <task‑title> → <new‑section>
- Added commit <hash>
```

This satisfies your existing requirement *“summary of the changes you made in the code”*.

---

## 7 · Example workflow

1. **Backlog** contains `- [ ] add‑login‑endpoint` (line 12).
2. Copilot starts coding ⇒ moves line 12 to **In Progress**.
3. Tests pass ⇒ moves to **Review / Testing** and adds commit hash.
4. You run your acceptance tests ⇒ Copilot moves it to **Done** with `[x]` and date.

---

## 8 · Quality gates

* No task may skip a stage.
* A PR cannot be merged while **In Progress** or **Review / Testing** lists are non‑empty.
* Linting, unit tests, and integration tests **must** be green before a task can be marked **Done**.

---

## 9 · House‑keeping

* Archive tasks older than 90 days in `archive/task‑YYYY‑MM.md`.
* Never delete history.

---

\### EOF
