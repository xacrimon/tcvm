---
name: review-changes
description: Senior-engineer code review of a change set — a PR, the branch vs main, or uncommitted work. Reviews for correctness, maintainability, duplication, inelegant code, and unjustified complexity, then presents a prioritized findings list and does not auto-fix. Use when asked to review changes, a PR, or the diff since the last commit.
argument-hint: "[pr-number | --staged | --uncommitted | <base-ref>]"
allowed-tools: Bash(git diff *) Bash(git log *) Bash(git status *) Bash(git show *) Bash(git merge-base *) Bash(git branch *) Bash(git rev-parse *) Bash(gh pr diff *) Bash(gh pr view *) Bash(cargo *)
---

# Review changes

Act as a senior engineer reviewing a change set. Lock the scope first, then review, then present findings. Do not fix anything during the review.

## Orientation
- Branch: !`git branch --show-current`
- Working tree: !`git status --short`
- Recent commits: !`git log --oneline -12`

## 1. Determine scope (always first)

If `$ARGUMENTS` is given:
- a number → GitHub PR: `gh pr diff <n>` and `gh pr view <n>`
- `--staged` → `git diff --cached`
- `--uncommitted` → `git diff HEAD`
- a branch/ref → `git diff <ref>...HEAD`

With no argument, infer from the orientation above:
- On a feature branch ahead of the default branch → review the whole branch via the merge-base: `git diff <default>...HEAD` (default is usually `main`). Read it commit-by-commit too, not just the squashed diff.
- Otherwise, if there are uncommitted changes → `git diff HEAD`.

If the scope is not extremely clear — e.g. both a branch ahead of main and uncommitted edits, or no obvious change set — ask which scope to review before proceeding. Don't guess.

## 2. Review

Read the full diff, plus surrounding code where needed, at two levels:
- Large scale: architecture, abstractions, duplication across files, things that will be painful to maintain.
- Line level: correctness bugs and edge cases, inelegant or convoluted code, dead or stale comments, over-commenting (per CLAUDE.md), performance costs that aren't worth it.

Verify before asserting (CLAUDE.md): never claim Lua / `luac` or other reference behavior without running it; build, test, or clippy when a finding depends on it. Cite `file:line`.

## 3. Present findings

Do not edit code. Present a prioritized list grouped by severity:
- Must-fix (correctness / bugs)
- Design / maintainability
- Minor / nits

Each item: a one-line problem statement, `file:line`, and a short why. Keep it tight. End by offering to fix or triage; for non-obvious design changes, discuss before deviating.
