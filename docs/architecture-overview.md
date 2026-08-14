# {System} architecture overview

_Also called: architecture document, solution architecture document (SAD)._

_Italic text is guidance. Delete it as you fill each section in._

|                    |                                                                            |
| ------------------ | -------------------------------------------------------------------------- |
| **Owner**          | _The team accountable for this document, not an individual who may leave._ |
| **Last reviewed**  | YYYY-MM-DD                                                                 |
| **Review cadence** | _Quarterly, or on every change that moves a boundary. State one._          |

_This document answers "what is this system and how is it shaped". It does not answer "why is it shaped that way": that belongs in [ADRs](architecture-decision-record.md), linked from here. Mixing the two produces a document that is neither consultable nor readable._

_Audience: a competent engineer who does not work on this system. A new hire in week one, an incident responder at 3am, a team that wants to integrate, an auditor. Write for them, not for yourself._

---

## 1. Purpose and scope

_What the system does, in business terms, in three sentences. Then what is inside the boundary and what is outside it._

_The boundary is the useful part. Most confusion about a system is confusion about where it ends._

> **Example.** Handles order capture, payment authorisation and fulfilment handoff for the retail site. Inventory and pricing are outside the boundary and owned by Catalogue.

---

## 2. Quality attributes

_The handful of non-functional properties this architecture is actually built to deliver, each with a number and a source._

_Every system claims to value availability, performance and security. That claim decides nothing. The value here is the ranking and the numbers, because they are what justify the structure below and what a reviewer can hold you to._

| Attribute      | Target                              | Why it matters                | Where it is measured |
| -------------- | ----------------------------------- | ----------------------------- | -------------------- |
| _Availability_ | _99.9% monthly for checkout_        | _Contractual_                 | _Link to SLO_        |
| _Latency_      | _p99 under 400ms for authorisation_ | _Cart abandonment above this_ | _Link to dashboard_  |

_Then, in one or two lines: which of these you trade away when they conflict. A system that ranks nothing has not made an architectural choice._

---

## 3. Constraints

_What was not open to choice, and by whom it was imposed. Technical, organisational, regulatory, contractual._

_A reader who does not know the constraints will read the architecture as a series of odd decisions. Name them once here and half the questions disappear._

- _Constraint, and its source: "Card data may not touch our infrastructure (PCI DSS scope reduction, mandated by Security)"_

---

## 4. Context

_The system as one box, and everything it talks to._

_Include: users and roles, external systems, and for each connection what flows, in which direction, over what protocol. Nothing about internals._

_Draw this as a diagram in a text format that can be diffed and reviewed. Include the source in the repo, not an exported image._

| Neighbour | Direction | What flows | Protocol | Owner |
| --------- | --------- | ---------- | -------- | ----- |
|           |           |            |          |       |

---

## 5. Containers

_The separately deployable or runnable things inside the boundary: services, jobs, databases, queues, front ends. What each is responsible for, what it is built with, and how the others reach it._

_Keep the responsibility to one sentence. If you cannot, that container does more than one thing, and saying so here is more useful than hiding it._

| Container | Responsibility | Technology | Called by | Calls |
| --------- | -------------- | ---------- | --------- | ----- |
|           |                |            |           |       |

_Then a diagram showing the containers and the connections between them._

_Stop here for most systems. Component-level detail below the container is worth documenting only where it is genuinely non-obvious, and it goes stale fastest._

---

## 6. Runtime behaviour

_Two to four scenarios traced end to end through the containers above. Pick the ones that matter: the main flow, the money flow, the flow that breaks most often._

_Numbered steps or a sequence diagram. Include what happens when a step fails, because that is what an incident responder came here to find._

### {Scenario name}

1. _..._
2. _..._

**When it fails.** _Where the retry is, what is idempotent, what a user sees._

---

## 7. Data

_What data lives where, which store is authoritative for each entity, and how data moves between them._

_The authoritative-source column is the point of this section. Systems rot when two stores both look authoritative for the same thing._

| Entity | System of record | Copies live in | How copies update | Retention |
| ------ | ---------------- | -------------- | ----------------- | --------- |
|        |                  |                |                   |           |

_Also note anything with legal or privacy weight: personal data, payment data, anything with a mandated retention or deletion period._

---

## 8. Deployment

_How the containers above map onto real infrastructure. Environments, regions, what is redundant and what is not._

_Say plainly where the single points of failure are. Every system has some. A document that implies there are none is not trusted by anyone who has been on-call._

---

## 9. Cross-cutting concerns

_The conventions that apply everywhere, stated once so no one has to infer them from code._

_Only the ones that are real for you. Delete the rest._

- **Authentication and authorisation.** _How a caller is identified and what decides access._
- **Error handling and retries.** _The house rules: what is retried, with what backoff, what is never retried._
- **Observability.** _Where logs, metrics and traces go, and what correlates them._
- **Configuration and secrets.** _Where they come from and how they rotate._
- **Versioning and compatibility.** _What guarantees you make to callers._

---

## 10. Architecture decisions

_Links to the ADRs that produced this structure, newest first. Not summaries: links._

_This section is what keeps the rest of the document short. Anywhere a reader is likely to ask "why like that", link the ADR instead of answering inline._

- [ADR-0007: Use Postgres for the ledger](decisions/0007-use-postgres-for-the-ledger.md)

---

## 11. Known problems and direction

_What is wrong with this architecture today, and what you intend to do about it. Debt with consequences, not a wish list._

_Every experienced reader already suspects the weak points. Naming them buys credibility for everything above, and it stops a new engineer proposing a fix you rejected two years ago._

| Problem | Impact today | Intended direction | Owner |
| ------- | ------------ | ------------------ | ----- |
|         |              |                    |       |

---

## 12. Glossary

_Domain terms used above, or a link to the [glossary](glossary.md). Define anything a competent outsider would guess wrong._

---

## Notes on using this template

_Delete this section too._

**Cut sections without hesitation.** This skeleton follows arc42's structure and C4's abstraction levels, and both are explicit that the sections are a checklist, not a mandatory contents page. A single service with one datastore needs sections 1, 4, 5 and 10 and nothing else.

**Keep diagrams as text.** Structurizr DSL, PlantUML, Mermaid, D2: anything that lives in the repo, diffs in review, and regenerates in CI. An exported PNG on a wiki cannot be reviewed alongside the change that invalidated it, and so it will not be.

**Give it an owner and a review date, or it will rot.** This is the document in the group most likely to be quietly wrong, because nothing fails when it drifts. The review cadence in the header is the only thing standing between it and fiction. If a review passes with no change, update the date anyway: "reviewed and still correct" is information.

**Where this lives:** in the repository, rendered to HTML in CI so non-engineers can read it without cloning.

---

## Related documents

- [`architecture-decision-record.md`](architecture-decision-record.md). The reasoning behind this structure; this document only shows the result
- [`service-readme.md`](service-readme.md). One level down, for a single deployable component
- [`glossary.md`](glossary.md). Domain terms used above, kept beside this document so both move together
- [`data-model.md`](data-model.md). Which service owns which entity, in more detail than section 7 carries
