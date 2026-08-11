# Security policy

| | |
|---|---|
| **Last reviewed** | 2026-08-11 |
| **Applies to** | This repository. The project ships no service and no released package. |

## Reporting a vulnerability

Use [private vulnerability reporting](https://github.com/alekszlat/Training-Dynamic-Gesture-Model/security/advisories/new). GitHub only notifies maintainers who are watching the repository, so if you have no reply in seven days, email alexzlat2005724@gmail.com or hristo.tsenkov.hristov@gmail.com.

Do **not** open a public issue or pull request for an unfixed vulnerability.

## What to include

Steps to reproduce or a proof of concept, the tools and versions you used, and their output. No CVSS score and no patch required.

## Our response

Two people maintain this in their spare time. Expect acknowledgement within seven days. You will be credited in the advisory unless you ask not to be.

## Scope

This is an offline training pipeline. It has no users, no accounts, and no network surface, and it processes only data the operator supplies.

In scope: anything here that lets a crafted video, manifest, or model file execute code or write outside `data/` on the machine running the pipeline.

Out of scope: flaws in MediaPipe, OpenCV, PyTorch, or any other dependency and anything that needs the attacker to already control the machine, the command line, or the input data.
