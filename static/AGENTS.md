# Static UI guidelines

This file applies to all frontend files under `static/`. Use it together with
the repository root `AGENTS.md`. If a rule here is more specific, it takes
precedence for this directory.

## Collaboration rules

Start from the user task, the review flow, and the actual operator goal. Do
not optimize for visual novelty first, and do not assume the screen target or
interaction path is already fully defined.

If the target page, the decision scenario, the core data meaning, or the
acceptance standard is unclear, stop and discuss before implementation. Do not
fill product gaps with self-invented UI logic.

When proposing frontend or UI changes, follow these constraints:

- Do not give patch-style plans that only cover symptoms.
- Do not overdesign. Use the shortest valid path that solves the real user
  problem.
- Do not add fallback, downgrade, or extra interaction branches unless the
  user explicitly asks for them.
- Ensure the full interaction chain is logically correct before coding, from
  user action to state update to final rendering.

## UI implementation constraints

Keep the current stack simple. This frontend is plain HTML, CSS, and
JavaScript. Do not introduce a new framework, build step, or component library
unless the user explicitly asks for that architectural change.

Prefer shared primitives over page-local duplication:

- Put reusable visual tokens and layout rules in `static/styles.css`.
- Put shared runtime, theme, toast, and common helper logic in `static/ui.js`.
- Keep page-specific orchestration in `static/app.js`, `static/admin.js`, and
  `static/eval.js`.

When changing UI presentation, update the whole visual system that supports the
task, not just one isolated control:

- Theme changes must affect background, surface, text, border, and emphasis
  contrast together. Do not leave theme switching at "button color only."
- Dense score panels, explanation cards, and detail areas must stay readable
  for non-expert reviewers.
- If content can overflow horizontally or vertically, the scroll behavior must
  remain directly reachable and natural.
- New labels should use direct Chinese wording when the target user is not
  expected to understand metric abbreviations or internal jargon.

Preserve existing task continuity:

- Do not break persisted runtime state stored in `localStorage`.
- Do not remove job lookup, task tracking, or page return continuity when
  redesigning a screen.
- If a task can continue running in the background, the UI must still let the
  user find that task again after navigation or refresh.

## Delivery checks

Before finishing a frontend change in `static/`, verify the following:

1. Check `static/index.html`, `static/admin.html`, or `static/eval.html` for
   the affected flow, depending on the page you touched.
2. Check both light and dark themes if the change affects layout, color,
   contrast, or emphasis.
3. Check narrow and wide viewport behavior if the change affects panels, tables,
   score cards, or sticky areas.
4. If you changed `static/styles.css`, `static/ui.js`, `static/app.js`,
   `static/admin.js`, or `static/eval.js`, sync the `?v=` query string in the
   referencing HTML file when needed so browser cache does not hide the update.
