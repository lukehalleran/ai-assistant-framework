# G09: Accessibility and Communication Generalization

## Objective

Make the Windows application and assistant interaction usable across supported
motor, visual, cognitive, learning, and communication needs while avoiding one
owner's vocabulary, cadence, emotional style, or literacy level as the default.

WCAG 2.2 AA is the minimum web-interface target. It is not sufficient evidence
for all cognitive or communication needs, so manual and external user testing is
required.

References:

- https://www.w3.org/TR/WCAG22/
- https://www.w3.org/WAI/cognitive/
- https://www.w3.org/TR/coga-usable/

## Supported communication variation

- Terse fragments and long-form messages.
- Typos, omitted punctuation, abbreviations, slang, and profanity.
- Regional, ethnic, generational, occupational, and social dialects.
- Code-switching where the primary interaction remains American English.
- Literal, indirect, humorous, ironic, or emotionally expressive writing.
- Custom names, pronouns, honorifics, and relationship terms.
- Users who prefer plain language, detailed explanations, or reduced stimulation.
- Keyboard, screen reader, magnification, high contrast, and speech-to-text input.

The system must not label a dialect as incorrect, less intelligent, or inherently
more distressed.

## Step-by-step plan

### G09-T01: Create an accessibility baseline

1. Inventory every route, control, dialog, stream, status message, toast, chart,
   document preview, onboarding step, and installer screen.
2. Run automated axe-style checks and record failures by WCAG criterion.
3. Test keyboard traversal, visible focus, zoom/reflow, Windows high contrast, and
   reduced motion manually.
4. Test NVDA with the primary chat, approval, settings, memory, and error paths.
5. Record inaccessible third-party components and replacement plans.

Acceptance:

- `G09-A01`: Every user-facing route has an accessibility inventory entry.
- `G09-A02`: No critical flow depends solely on pointer input, hover, color, or
  visual position.

### G09-T02: Make chat streaming accessible

1. Give messages, sources, tool progress, errors, and action proposals semantic
   regions and labels.
2. Do not announce every streamed token to screen readers.
3. Buffer announcements into meaningful updates and let users pause them.
4. Preserve focus when messages, citations, or action cards appear.
5. Provide stop-generation and retry controls reachable by keyboard.
6. Distinguish assistant response, retrieved quotation, system status, and error
   without relying only on color.

Acceptance:

- `G09-A03`: NVDA can follow a complete streamed turn without token-by-token
  interruption or lost focus.
- `G09-A04`: Stop, retry, approve, reject, and inspect-source actions work without
  a mouse.

### G09-T03: Fix structure, focus, and input behavior

1. Use one logical heading hierarchy and named landmarks.
2. Give every field a persistent programmatic label and clear error association.
3. Move focus deliberately on dialogs and return it on close.
4. Eliminate keyboard traps and hidden-focus targets.
5. Support browser/OS zoom to at least 200 percent and reflow at narrow widths.
6. Respect text-spacing overrides and high-contrast colors.
7. Ensure target sizes and spacing meet WCAG 2.2 AA.

Acceptance:

- `G09-A05`: Automated keyboard scripts traverse each critical workflow in a
  stable order.
- `G09-A06`: At 200 percent zoom, no required content or action is lost or
  overlapped.

### G09-T04: Make time and motion user controlled

1. Avoid expiring approvals while a user is actively reviewing them; warn and
   allow extension where security permits.
2. Let users pause auto-updating diagnostics and long-running activity streams.
3. Honor reduced-motion settings.
4. Never require a rapid multi-step gesture.
5. Preserve form and chat input across recoverable errors.

Acceptance:

- `G09-A07`: Core tasks have no unadjustable user-facing time limit.
- `G09-A08`: Reduced-motion mode removes nonessential animation without hiding
  state changes.

### G09-T05: Add communication preferences

1. Separate warmth, directness, verbosity, plain-language level, structure, and
   proactive behavior into independent preferences.
2. Permit temporary per-turn overrides.
3. Let users disable emotional mirroring and proactive emotional inference.
4. Make preference changes previewable and reversible.
5. Avoid equating concise language with coldness or long language with expertise.
6. Store these as declared preferences, not inferred protected attributes.

Acceptance:

- `G09-A09`: Preference combinations produce distinct but semantically equivalent
  answers on the frozen style suite.
- `G09-A10`: Style settings never weaken factual, safety, privacy, or action
  requirements.

### G09-T06: Build plain-language response support

1. Add a user-controlled plain-language instruction independent of intelligence
   or literacy labels.
2. Prefer common words, short sections, explicit referents, and one instruction
   per step when enabled.
3. Preserve necessary technical terms with definitions rather than deleting
   substance.
4. Avoid unexplained metaphors, ambiguous pronouns, nested negatives, and dense
   disclaimer blocks.
5. Offer a concise summary for long generated documents.

Acceptance:

- `G09-A11`: Human review confirms plain-language output preserves required facts
  and actions.
- `G09-A12`: Readability changes cannot alter tool arguments or safety policy.

### G09-T07: Generalize language routing

1. Audit regexes, keywords, length gates, and exemplars for dialect and owner
   assumptions.
2. Build invariant tests across paraphrase, spelling, punctuation, verbosity,
   and dialect variation.
3. Include non-triggering counterexamples for irony, quotations, gaming, news,
   and third-person descriptions.
4. Prefer calibrated uncertainty and clarification over confident misrouting.
5. Measure intent/tone false positives and negatives by communication subgroup.
6. Do not normalize the user's words before preserving the source excerpt.

Acceptance:

- `G09-A13`: High-impact routing meets G05 worst-group gates.
- `G09-A14`: Dialect markers alone do not trigger distress, refusal, lower
  helpfulness, or unnecessary tools.

### G09-T08: Make onboarding accessible and optional

1. Replace conversational-only onboarding with a structured accessible path while
   retaining optional conversational help.
2. Explain why each profile field is requested and permit skip.
3. Avoid requiring a name, binary pronoun set, location, integrations, or personal
   history.
4. Let users test speech-to-text and screen-reader behavior before long setup.
5. Show disk, model, privacy, and network consequences in plain language.
6. Save progress and allow later completion.

Acceptance:

- `G09-A15`: A keyboard/NVDA user can install and onboard without developer help.
- `G09-A16`: Skipping every optional identity field yields a functional product.

### G09-T09: Make errors and recovery accessible

1. Place errors near the affected control and summarize them in an announced
   region.
2. Explain what happened, what remains safe, and the next concrete action.
3. Preserve user input when retry is possible.
4. Avoid raw provider/runtime traces in ordinary UI.
5. Make backup, migration, and model-load progress understandable without visual
   monitoring.

Acceptance:

- `G09-A17`: Screen-reader users receive one useful error announcement rather
  than repeated streaming noise.
- `G09-A18`: Recovery instructions work in offline mode.

### G09-T10: Add automated accessibility gates

1. Run automated accessibility checks in frontend CI.
2. Use Playwright for keyboard order, focus restoration, zoom/reflow, high
   contrast, reduced motion, and streamed-content behavior.
3. Test long names, long unbroken strings, large text, and translated-like
   expansion even though 1.0 is American English.
4. Capture screenshots at supported scale factors for overlap review.
5. Treat critical-flow accessibility regressions as release blockers.

Acceptance:

- `G09-A19`: Stable release has no known WCAG A/AA violation in a critical flow.
- `G09-A20`: Screenshot and DOM checks show no clipped or occluded required text.

### G09-T11: Conduct external accessibility testing

1. Recruit users of screen readers, magnification, keyboard-only navigation,
   speech input, and cognitive/learning accommodations.
2. Compensate structured testing.
3. Test installer, onboarding, chat, memory correction/deletion, permissions,
   settings, and recovery.
4. Record task success, errors, time, and qualitative barriers.
5. Retest confirmed fixes with affected users.

Acceptance:

- `G09-A21`: Stable 1.0 has manual external evidence for the declared assistive
  technology matrix.
- `G09-A22`: Known unsupported needs are documented rather than generalized away.

## Assistive technology matrix for Windows 1.0

- NVDA with current supported Firefox and Chromium-based browser/UI shell.
- Keyboard-only navigation.
- Windows high contrast.
- 200 percent text zoom and high DPI scaling.
- Reduced motion.
- Windows speech-to-text input.
- Plain-language and concise-output preferences.

Additional tools may be supported after evidence exists.

## Exit gate

G09 is validated when critical workflows meet WCAG 2.2 AA, automated checks run
in CI, the declared Windows assistive-technology matrix passes manual external
testing, communication preferences are independent and safe, and language
routing meets population worst-group gates.

