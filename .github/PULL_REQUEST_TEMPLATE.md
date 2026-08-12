<!--
Title in the imperative, as if completing "this change will…" — "Cache the resolved
config", not "Caching fix". The title usually becomes the commit message.
-->

Fixes #

<!-- Use Fixes/Closes only if merging should close the issue. Otherwise: Refs #123. -->

## Summary

<!--
Why this change exists, then what it does. Motivation first.

Do not describe the diff — a reviewer can read the diff. Describe what it cannot
say: why now, what you ruled out, and what you are still unsure about.
-->

## Testing

<!--
What you ran and what it covered, specific enough that a reviewer could repeat it.
"Tested locally" tells them nothing.

Say what you did not test, so it can be checked by someone else, rather than commited.
-->

## Breaking changes

<!--
Delete this section if nothing breaks.
Otherwise: what stops working, who it affects, and the exact steps to move across.
-->

## Release note

<!--
One sentence for the changelog, written for someone who does not know this codebase
— or "none".

Good: Fix a panic when running `docker top` on a non-running Windows container.
Bad:  Refactor test TestFooWithBar

The bad one describes the work rather than the user-visible effect.
-->

## AI assistance

<!--
If you used an AI tool on this change, say which one and what you did to check its
output. Use is not banned; verification stays with you.
-->

## Checklist

- [ ] Tests cover the change
- [ ] Documentation updated
- [ ] No unrelated changes included
