# ADR-NNNN: {Short title stating the decision}

_Also called: Architecture Decision Record (ADR), decision record._

_Italic text is guidance. Delete it as you fill each section in._

_Title the decision, not the topic. "Use Postgres for the ledger" beats "Database choice". A reader scanning a list of forty ADRs should learn the outcome from the title alone. Number files sequentially and never renumber: `0007-use-postgres-for-the-ledger.md`._

|              |                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------ |
| **Status**   | Proposed \| Accepted \| Rejected \| Deprecated \| Superseded by [ADR-0012](0012-{slug}.md) |
| **Date**     | YYYY-MM-DD (the date the status last changed)                                              |
| **Deciders** | Names or a role, whoever actually decided                                                  |

_Never edit an accepted ADR to change its decision. Write a new one and mark the old one `Superseded by`. The point of the log is that it shows how thinking changed. An ADR you can rewrite is a wiki page with extra steps._

---

## Context and problem statement

_The forces in play, told neutrally. What is true today, what pressure made this a question now, what constraints are fixed. Written so someone who joins in two years understands the situation without asking anyone._

_Include the constraints that turn out to be decisive: a compliance requirement, a team's existing skills, a contract, a latency budget, a deadline. Those are exactly what code cannot show and what people forget first._

_State it as a problem, not a solution. If this section names your preferred option, you have written the conclusion in the wrong place._

_Two or three paragraphs. If it runs longer, the decision is probably several decisions._

> **Example.** Orders are written to the same MySQL instance that serves the catalogue. Catalogue reads spike at 40x during campaigns and have twice caused order writes to time out. The ledger must survive audit, so writes need durability guarantees we cannot relax. We have no DBA; whatever we pick, the four of us operate it.

---

## Decision drivers

_The criteria you judged options against, listed before the options so the ranking is visible. Optional, but it stops the classic failure where the "Considered options" section quietly rewards whatever was already chosen._

_Order them. Unordered criteria decide nothing._

- _Driver 1 (for example: no new operational surface the team cannot run)_
- _Driver 2_
- _Driver 3_

---

## Considered options

_A flat list of the options, one line each. Detail goes further down. Two options is the minimum that makes an ADR worth writing; "do nothing" is a legitimate option and often the strongest._

1. _Option A_
2. _Option B_
3. _Option C_

---

## Decision

_One sentence, active voice, present tense: "We will ...". Then a short paragraph on why this option beat the others against the drivers above._

_Do not hedge. An ADR that says the team "leans towards" something has not recorded a decision, and the next reader will not know whether they are allowed to build on it._

> **Example.** We will move the order ledger to a dedicated Postgres instance, keeping the catalogue on MySQL. Postgres gives us the transactional guarantees the audit needs, and a separate instance removes the contention that caused both outages. We accept running two engines because splitting the load matters more than reducing the number of technologies.

---

## Consequences

_What becomes true once this is done. Both directions, honestly._

_The negative list is the one that earns the document. An ADR with only benefits reads as advocacy and gets distrusted. Naming the costs up front is also what lets a later team recognise when a cost grew past what you accepted._

**Positive**

- _What gets easier, faster, safer, cheaper._

**Negative**

- _What gets harder, slower, more expensive, or more fragile. Include the ones you are choosing to accept._

**Follow-on work**

- _Anything this decision forces someone to do, with an owner. Migration, a new alert, a deprecation, a document to update._

> **Example (negative).** Two database engines to patch, back up and monitor. Cross-entity reporting queries can no longer be a single join and will need the reporting pipeline instead.

---

## Options in detail

_Optional. Include it when the rejected options were genuinely close, or when someone will propose them again. Skip it when the choice was obvious once the constraints were written down._

_Pros and cons only. No narrative. The value here is that a future engineer can check whether your reasoning still holds after the constraints change._

### Option A: {name}

- **Good.** _..._
- **Good.** _..._
- **Bad.** _..._
- **Neutral.** _..._

### Option B: {name}

- **Good.** _..._
- **Bad.** _..._

---

## Confirmation

_How anyone can verify the decision was actually carried out. A test, a lint rule, a CI check, an architecture fitness function, a review step. One line._

_Decisions with no confirmation quietly stop being true. This section is what separates a record from an intention._

> **Example.** CI fails if any module under `orders/` imports the MySQL client.

---

## More information

_Links: the design doc, the spike, the benchmark, the incident that triggered this, the ADRs this one relates to. Anything a reader would otherwise have to hunt for._

_Also record the expiry condition if one exists: "revisit if write volume exceeds 5k/s" tells a future team when to reopen this rather than leaving them to guess._

---

## Notes on using this template

_Delete this section too._

**Keep it to one or two pages.** Nygard's original argument was that short records get written and long ones do not. If a decision needs twenty pages, that belongs in a design document, and the ADR records the outcome and links to it.

**Write it when the decision is made, not after the code ships.** An ADR written retroactively records what you did, which the code already shows. Written at decision time it records what you rejected, which nothing else does.

**Sections you may cut:** decision drivers, options in detail, confirmation. Sections you may not: context, decision, consequences. Those three are Nygard's minimum and every later format keeps them.

**Where this lives:** `docs/decisions/` in the repository that the decision constrains. Decisions spanning several repositories go in a central log, linked from each.

---
