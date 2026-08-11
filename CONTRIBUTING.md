# Contributions
Thank you for your interest in contributing to this project!

This guide provides an overview of the contribution process, including how to report issues, propose improvements, submit changes, and work with the project's development workflow. It also outlines the conventions and expectations contributors should follow when making changes to the repository.

**Please read this guide before submitting an issue or pull request.**

For what the project is and how to run it, see [`README.md`](README.md).

## Where to ask

There is no chat channel. Open an [issue](https://github.com/alekszlat/Training-Dynamic-Gesture-Model/issues/new/choose), or comment on the one you are working from. Questions belong in the tracker while the project is this small.

## What counts as a contribution

Code, documentation, bug reports, recorded gesture data, and reviews of open pull requests.

## Reporting a bug

Search the [open issues](https://github.com/alekszlat/Training-Dynamic-Gesture-Model/issues) first, then use the **Bug report** form. Title it by the effect you saw, not the cause you suspect.

A security vulnerability does not go in the tracker. Follow the [security policy](.github/SECURITY.md) instead.

## Proposing a change

**Anything beyond a typo or an obvious fix needs an issue before a pull request. Pull requests without one may be closed.** The scope of the project is still being settled, so agreeing on the problem is cheaper than reworking a finished change.

Use the **Feature request** form. Describe what you were trying to do, not the implementation you have in mind.

## Workflow

The project follows [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow): one permanent branch, short-lived branches off it, and every change merged back through a pull request. `main` is the single source of truth and is kept in a working state.

A change moves through four stages:

1. **Issue.** Every change starts as one, on the bug report or feature request form.
2. **Discussion.** Scope, approach, and whether to do it at all are settled in the issue thread before code is written.
3. **Trello.** Accepted issues become cards on the project board. The tracker holds *what and why*; the board holds *where it is now*.
4. **Pull request.** Branch, implement, open it against `main`, get it reviewed, squash-merge.

Branch from the latest `main` and name it `<type>/<short-description>`:

| Prefix | For |
|---|---|
| `feature/` | new functionality |
| `fix/` | bug fixes |
| `docs/` | documentation |
| `chore/` | maintenance, dependencies, configuration |

Branch prefixes are spelled out (`feature/`), commit types are not (`feat:`).

- A branch lives two or three days. If the change is bigger than that, split it: long branches diverge, and the merge pays for it.
- Nothing reaches `main` except through a pull request, even while nothing enforces it.
- Pull requests are squash-merged, so the title becomes the commit message. Delete the branch afterwards.

## Submitting a pull request

1. One pull request, one logical change.
2. Commit with [conventional commits](https://www.conventionalcommits.org): `<type>(<scope>): <subject>`, imperative, lower case, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `build`, `ci`, `chore`, `perf`, `style`.
3. Run `uv run pytest` before opening. There is no CI yet, so the checks you run are the only ones that run.
4. Fill in the pull request template, including what you did **not** test.

The pipeline scripts (`01_`–`03_`) run in order and only from the repository root.

## What not to do

- Do not commit datasets, generated tensors, or `hand_landmarker.task`. They are ignored deliberately.
- Do not force-push a branch someone else may be reading.
- Do not touch files outside the scope of the change. Note anything unrelated in a separate issue.

## AI-assisted contributions

Permitted, on these terms:

- Say in the pull request which tool you used.
- Review the output yourself before asking a human to review it.
- Do not credit a tool as author, co-author, or committer — no `Co-Authored-By` or `Assisted-by` trailers.
- Answer review comments yourself.
- Keep at most one AI-assisted pull request open at a time.

## Licence

The repository has no `LICENSE` yet ([#1](https://github.com/alekszlat/Training-Dynamic-Gesture-Model/issues/1)), so default copyright applies. By opening a pull request you agree your contribution will fall under the licence the project adopts.
