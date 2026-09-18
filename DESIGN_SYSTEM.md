# AgentForce Design System — "Tasko" Visual Language

Reverse-engineered from the Tasko reference dashboard
(https://v0-dashboard-ui-redesign-nine.vercel.app/ — shadcn/ui + Tailwind v4 + Recharts + Geist),
adapted to AgentForce's actual stack: **Python-generated, self-contained HTML**
(`src/agentforce/visualization.py`) with inline CSS under a strict CSP
(`style-src 'unsafe-inline'; img-src data:` — no external fonts, scripts, or stylesheets).

Two forms are given for every rule:
- **Tailwind classes** — the reference implementation (useful if a JS frontend is ever added).
- **CSS variables / plain CSS** — what AgentForce actually uses; tokens live in
  [`src/agentforce/assets/design_tokens.css`](src/agentforce/assets/design_tokens.css) and are inlined at render time.

Goal: reproduce the *visual language* (deep green primary, soft off-white surfaces, large radii,
generous whitespace, gentle motion) — not the Tasko content.

---

## 1. Color palette

Semantic tokens (extracted verbatim from the reference's `:root`):

| Token | Light | Dark | Usage |
|---|---|---|---|
| `--background` | `#f8f9f5` | `#040705` | App/page background (warm off-white / near-black green) |
| `--foreground` | `#202318` | `#f6f9f7` | Default text |
| `--card` | `#ffffff` | `#080c09` | Card & sidebar surface |
| `--card-foreground` | `#202318` | `#f6f9f7` | Text on cards |
| `--popover` | `#ffffff` | `#080c09` | Popovers/menus |
| `--primary` | `#005e30` | `#008b46` | Brand deep green — active nav, primary buttons, hero stat card |
| `--primary-foreground` | `#f3fbf5` | `#f3fbf5` | Text/icons on primary |
| `--secondary` | `#f1f3eb` | `#0f1912` | Subtle fills, hover background |
| `--secondary-foreground` | `#202318` | `#f6f9f7` | Text on secondary |
| `--muted` | `#f1f3eb` | `#0f1912` | Muted fills, skeletons |
| `--muted-foreground` | `#707367` | `#86938a` | Secondary text, labels, captions |
| `--accent` | `#008b46` | `#008b46` | Bright green — links, highlights, "peak" values |
| `--accent-foreground` | `#f3fbf5` | `#f3fbf5` | Text on accent |
| `--destructive` | `#e40014` | `#bb061e` | Errors, notification dots |
| `--destructive-foreground` | `#fff6f8` | `#fff6f8` | Text on destructive |
| `--border` | `#e3e6de` | `#1a251d` | Hairline borders |
| `--input` | `#e3e6de` | `#1a251d` | Input borders |
| `--ring` | `#005e30` | `#008b46` | Focus rings |

Chart ramp (same in both modes — a 5-step green scale):

```
--chart-1: #005e30   /* darkest — primary series */
--chart-2: #008b46
--chart-3: #49a46e
--chart-4: #85bd98
--chart-5: #bad6c3   /* lightest */
```

Status colors (pastel chip + strong text — Tailwind `*-100` / `*-700`):

| Status | Background | Text | Tailwind |
|---|---|---|---|
| Success / Completed / exact match | `#d1fae5` | `#047857` | `bg-emerald-100 text-emerald-700` |
| Warning / In Progress / nearest match | `#fef3c7` | `#b45309` | `bg-amber-100 text-amber-700` |
| Danger / Pending / missing | `#ffe4e6` | `#be123c` | `bg-rose-100 text-rose-700` |

AgentForce modality accents (kept from current UI, re-tuned to sit with the green palette):

```
--mod-visual: #7c5bd0   --mod-asr: #047857   --mod-ocr: #b45309   --mod-objects: #0e7490
```

Rule: **one saturated hue (green) does all the branding work**; everything else is neutral
warm-gray-green. Never introduce a second saturated brand color.

## 2. Typography scale

Only these steps exist (Tailwind names → px/weight as computed on the reference):

| Role | Tailwind | CSS |
|---|---|---|
| Micro label / badge / section header | `text-[10px] font-medium uppercase tracking-wider` | `10px / 500 / letter-spacing .05em / uppercase` |
| Caption, meta, badges | `text-xs` (`font-medium` in chips) | `12px / 400–500 / lh 16px` |
| Body, buttons, inputs, nav items | `text-sm font-medium` | `14px / 500 / lh 20px` |
| Emphasized body | `text-base font-semibold` | `16px / 600` |
| Card sub-title | `text-lg font-semibold` | `18px / 600` |
| Card title | `text-xl font-semibold` | `20px / 600 / lh 28px` |
| Page title (responsive) | `text-xl md:text-2xl lg:text-3xl font-bold` | `20→24→30px / 700 / lh 1.2` |
| Stat number | `text-3xl font-bold` | `30px / 700 / lh 36px` |
| Hero number (donut %, timer) | `text-4xl` … `sm:text-5xl font-bold` | `36–48px / 700` |

Weights used: **500, 600, 700 only** (400 for plain body). No thin/light weights, no italics.
Page subtitle pattern: `text-sm text-muted-foreground` directly under the bold title (`mb-1` on title).

## 3. Font family

The reference ships **Geist** (variable 100–900) + **Geist Mono** via `@font-face`, falling back to
the system stack. Under AgentForce's CSP external fonts are impossible, so use the fallback stack
(identical rendering intent, zero bytes):

```css
--font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif,
             "Apple Color Emoji", "Segoe UI Emoji";
--font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
```

`body { font-family: var(--font-sans); -webkit-font-smoothing: antialiased; line-height: 1.45; }`
Mono is used only for scores, frame indexes, timestamps and raw evidence (`pre`).
(Optional Phase-2 upgrade: embed Geist as a `data:` URI and add `font-src data:` to the CSP.)

## 4. Spacing system

Standard 4px base scale; the reference is *tight but airy* — small paddings, consistent gaps:

| Token | Value | Tailwind | Where it's used |
|---|---|---|---|
| `--space-1` | 4px | `p-1 / gap-1` | icon nudges |
| `--space-1.5` | 6px | `px-1.5 py-1.5 gap-1.5` | badge padding, tiny gaps |
| `--space-2` | 8px | `py-2 gap-2` | nav item padding-y, button gap |
| `--space-2.5` | 10px | `px-2.5 gap-2.5` | nav item padding-x, icon–label gap |
| `--space-3` | 12px | `p-3 gap-3` | compact card padding, stat grid gap, main padding (mobile) |
| `--space-4` | 16px | `p-4 gap-4` | **default card padding**, sidebar padding, grid gaps |
| `--space-5` | 20px | `lg:p-5` | main content padding (desktop) |
| `--space-6` | 24px | `p-6 gap-6` | roomy card padding, card internal column gap |

Vertical rhythm: sections separated with `space-y-3 md:space-y-4`; title→subtitle `mb-1`;
card title→content `mb-3`/`mb-4`; page header block→content `mb-6`.

## 5. Border radius system

Global base **`--radius: 1rem` (16px)** — the signature "soft" look. Derived (shadcn v4 formula):

```css
--radius-sm: calc(var(--radius) - 4px);  /* 12px — small chips, pre blocks   */
--radius-md: calc(var(--radius) - 2px);  /* 14px — buttons, inputs           */
--radius-lg: var(--radius);              /* 16px — nav items, inner panels   */
--radius-xl: calc(var(--radius) + 4px);  /* 20px — cards (default surface)   */
--radius-full: 9999px;                   /* pills, badges, avatars, icon chips */
```

Tailwind: `rounded-md` buttons/inputs, `rounded-lg` nav items, `rounded-xl` all cards,
`rounded-full` badges/avatars/dots. **Nothing is square; nothing is under 12px** except chart bars.

## 6. Shadow system

Tailwind defaults + brand-tinted glow for primary elements:

```css
--shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);                                /* inputs, outline buttons */
--shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.10), 0 1px 2px -1px rgb(0 0 0 / 0.10); /* resting cards        */
--shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.10), 0 2px 4px -2px rgb(0 0 0 / 0.10);
--shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.10), 0 4px 6px -4px rgb(0 0 0 / 0.10); /* hero/stat cards, hover */
--shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.10), 0 8px 10px -6px rgb(0 0 0 / 0.10); /* card hover      */
--shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);                          /* dark feature card hover */
--shadow-primary: 0 10px 15px -3px rgb(0 94 48 / 0.20);                     /* active nav "glow"      */
```

Pattern: rest at `shadow-sm`, elevate on hover to `shadow-lg/xl`; the active nav item and primary
CTAs get the green-tinted `shadow-primary/20`. Borders and shadows are used *together* (1px border
+ soft shadow), never shadow alone.

## 7. Layout / grid rules

- App shell: fixed sidebar `w-64` (256px) + content `flex-1 p-3 md:p-4 lg:p-5 lg:ml-64`.
- Max content width: fluid (no max-w on the reference); AgentForce keeps `max-width: 1800px; margin: auto`.
- Stat row: `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3`.
- Main area: `grid grid-cols-1 lg:grid-cols-3 gap-3 md:gap-4` — dominant panel spans `lg:col-span-2`.
- Card galleries: `grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 md:gap-4`.
- CSS equivalent: `display:grid; grid-template-columns:repeat(N,minmax(0,1fr)); gap:12–16px; align-items:start`.
- Whitespace does the separating; **no horizontal rules between page sections**.

## 8. Sidebar / navigation rules

- Container: `fixed top-0 left-0 w-64 h-screen bg-card border-r border-border p-4 overflow-y-auto`,
  hidden below `lg` (`hidden lg:block`, hamburger `lg:hidden` in the top bar).
- Brand row: 32px `rounded-full bg-primary` logo chip + `text-lg font-bold` wordmark.
- Section labels: `text-[10px] font-medium text-muted-foreground uppercase tracking-wider mb-2`
  ("MENU", "GENERAL").
- Item: `w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-sm font-medium
  transition-all duration-300` with a 16px lucide-style stroke icon.
  - Inactive: `text-muted-foreground hover:bg-secondary hover:text-foreground`
  - **Active: `bg-primary text-primary-foreground shadow-lg shadow-primary/20`** (filled pill — the
    signature move)
  - Count badge: `ml-auto bg-primary text-primary-foreground text-[10px] font-semibold px-1.5 py-0.5 rounded-full`
    (on active item: `bg-primary-foreground/20`).
- Groups: MENU (primary destinations) then GENERAL (settings/help/logout), `space-y-0.5` items.

## 9. Card styles

Base (shadcn `Card`):

```
bg-card text-card-foreground flex flex-col gap-6 rounded-xl border shadow-sm p-4 (or p-6)
transition-all duration-500 hover:shadow-xl
```

CSS: `background:var(--card); border:1px solid var(--border); border-radius:var(--radius-xl);
box-shadow:var(--shadow-sm); padding:16px; transition:box-shadow .5s, transform .5s`.

Variants:
- **Hero/emphasis card** (first stat): same but `bg-primary text-primary-foreground shadow-lg` —
  invert one card per group to create hierarchy.
- **Inverted feature card** (Time-Tracker style): `bg-foreground text-background` with decorative
  blurred blobs, `hover:shadow-2xl`.
- **Gradient wash** (chart card): `bg-gradient-to-br from-background to-muted/20`.
- Card header: title `text-xl font-semibold` left, action/legend right, `flex items-start justify-between mb-4`.
- Entrance: `animate-slide-in-up` with staggered `animation-delay: index * 100ms`.

## 10. Button styles

Base (all variants): `inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md
text-sm font-medium h-9 px-4 py-2 transition-all outline-none disabled:opacity-50
disabled:pointer-events-none [&_svg]:size-4` → CSS: `height:36px; padding:8px 16px;
border-radius:var(--radius-md); font:500 14px var(--font-sans); gap:8px; cursor:pointer`.

| Variant | Classes / CSS |
|---|---|
| Primary | `bg-primary text-primary-foreground hover:bg-primary/90 shadow-lg hover:shadow-primary/30 hover:scale-105` |
| Outline | `border border-input bg-background shadow-xs hover:bg-accent hover:text-accent-foreground` (AgentForce: hover `background:var(--secondary)`) |
| Ghost | `hover:bg-secondary hover:text-foreground` (used for icon buttons, pagination) |
| Destructive | `bg-destructive text-destructive-foreground hover:bg-destructive/90` |
| Icon | `size-9 rounded-md` (36×36, icon 16px) |
| Small | `h-8 px-3 text-xs rounded-md` |

Full-width CTA inside cards (Start-Meeting style): primary + `w-full rounded-md` with leading icon.

## 11. Input / form styles

- Field: `h-9 w-full rounded-md border border-input bg-card px-3 py-1 text-sm shadow-xs
  placeholder:text-muted-foreground outline-none transition-[color,box-shadow]`
  → CSS: `height:36px; border-radius:14px; border:1px solid var(--input);
  background:var(--card); padding:4px 12px; font-size:14px; box-shadow:var(--shadow-xs)`.
- Focus: `focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]`
  → `border-color:var(--ring); box-shadow:0 0 0 3px color-mix(in srgb, var(--ring) 50%, transparent)`.
- Search pattern: relative wrapper, 16px search icon absolutely at left (`pl-9`), optional
  kbd hint chip at right (`⌘F` — `text-[10px] border rounded px-1.5 py-0.5 bg-muted text-muted-foreground`).
- Select: same box as input + chevron-down 16px at right; AgentForce styles native `<select>`
  identically (`appearance:none` + inline SVG chevron background is optional).
- Labels: `text-sm font-medium mb-1.5`; helper/error: `text-xs text-muted-foreground` / `text-destructive`.

## 12. Badge styles

| Kind | Classes / CSS |
|---|---|
| Status chip | `text-xs font-medium px-3 py-1.5 rounded-full whitespace-nowrap` + status colors from §1 (e.g. `bg-emerald-100 text-emerald-700`) |
| Count badge | `text-[10px] font-semibold px-1.5 py-0.5 rounded-full bg-primary text-primary-foreground` |
| Outline/tag badge (modality, objects) | `text-[10px] px-1.5 py-0.5 rounded-full border border-border text-muted-foreground` + modality accent color |
| Notification dot | `w-1.5 h-1.5 rounded-full bg-destructive absolute top-1.5 right-1.5 animate-pulse` |
| Legend dot | `w-2.5 h-2.5 rounded-full` + chart color, `gap-2` from its `text-xs text-muted-foreground` label |

Rule: badges are **always pill-shaped** (`rounded-full`), never squared.

## 13. Table styles

The reference renders lists as **borderless rows inside cards**, not chrome-heavy tables:

- Row: `flex items-center justify-between gap-3 p-3 rounded-xl border border-border bg-card
  transition-all duration-300 hover:shadow-lg hover:scale-[1.02]` — or borderless with
  `border-b border-border last:border-0` inside a card.
- Leading: 40–48px `rounded-full` avatar/emoji chip (`w-12 h-12 rounded-full bg-muted grid place-items-center`).
- Primary line `text-sm font-semibold`, secondary line `text-xs text-muted-foreground`.
- Trailing: status pill (§12) or action icon button.
- True data tables (modality scores): `width:100%; border-collapse:collapse; font-size:12px;`
  header `text-[10px] uppercase tracking-wider text-muted-foreground text-left pb-2`;
  cells `padding:6px 8px; border-bottom:1px solid var(--border)`; last row borderless;
  numeric cells `font-family:var(--font-mono)`.

## 14. Chart styles

(Reference uses Recharts; AgentForce reproduces with plain SVG/CSS.)

- Series colors come **only** from `--chart-1…5`; single-series bar charts alternate
  `#49a46e` (light) / dark `#005e30`-`#2d5540` bars to add rhythm.
- Bars: heavily rounded tops — `radius={[12,12,0,0]}` (12px corner radius), wide bars, `gap ~30%`.
- Grid: horizontal dashed hairlines only (`stroke:var(--border); stroke-dasharray:3 3`), no vertical lines, no axis lines/ticks.
- Axis labels: `12–14px`, `fill:var(--muted-foreground)`.
- Donut/radial: thick stroke (~12–16px), `stroke-linecap:round`, track in `--muted` or a
  diagonal-stripe pattern (`repeating-linear-gradient(45deg, var(--chart-5) 0 4px, transparent 4px 8px)`),
  big center number `text-4xl font-bold` + `text-xs text-muted-foreground` caption.
- Legend: dot + label chips (§12), placed top-right of card header.
- Chart entrance: `barSlideUp` (scaleY from 0, `transform-origin:bottom`, staggered 60–100ms).
- Footer stats row: `Average: **62%** … Peak: **92%**` — `text-sm text-muted-foreground` with
  `font-semibold text-foreground` (accent green for the highlight number).

## 15. Modal / dialog styles

(None open by default in the reference; these are the shadcn defaults its code ships with.)

- Overlay: `fixed inset-0 z-50 bg-black/50` + `fade-in 150ms`.
- Panel: `bg-card rounded-xl border shadow-lg p-6 w-full max-w-lg gap-4`
  (`--radius-xl`, `--shadow-lg`), entrance `zoom-in-95 + fade` 200ms ease-out.
- Title `text-lg font-semibold`, description `text-sm text-muted-foreground`.
- Footer: right-aligned `flex justify-end gap-2` — ghost/outline "Cancel" + primary confirm.
- Close: ghost icon button `size-8 rounded-md absolute top-4 right-4 opacity-70 hover:opacity-100`.
- AgentForce equivalent: `<details>`/`<dialog>` styled with the same tokens (CSP allows inline JS).

## 16. Hover / focus / active states

| State | Rule |
|---|---|
| Card hover | `hover:shadow-xl` (+ interactive rows `hover:scale-[1.02]`), `transition-all duration-300–500` |
| Button hover | fill variants → `/90` opacity of bg; ghost/outline → `bg-secondary`; primary CTAs may add `hover:scale-105 hover:shadow-primary/30` |
| Nav hover | `hover:bg-secondary hover:text-foreground` |
| Icon/avatar hover | `hover:scale-110`, avatar ring `ring-2 ring-primary/20 → hover:ring-primary/40`, playful `group-hover:rotate-12` on decorative icons |
| Focus (all interactive) | `focus-visible:border-ring focus-visible:ring-[3px] ring-ring/50` — 3px half-transparent green halo; never remove outlines without replacement |
| Active/pressed | scale back to 1.0 (`active:scale-100`), or `/80` bg |
| Disabled | `disabled:opacity-50 disabled:pointer-events-none` |
| Selection | `selection:bg-primary selection:text-primary-foreground` |

## 17. Dark mode rules

- Strategy: class toggle (`.dark`) in the reference; AgentForce keeps
  `@media (prefers-color-scheme: dark)` **and** supports a `.dark`/`.light` class override
  (`color-scheme: light dark` stays on `:root`).
- Only the semantic tokens flip (§1); component CSS never references raw hex.
- Dark surfaces are **near-black greens** (`#040705`, `#080c09`), not gray; borders `#1a251d`.
- `--primary` brightens `#005e30 → #008b46` (contrast on dark); chart ramp stays identical.
- Status chips in dark mode: use `color-mix(in srgb, <color-700> 25%, transparent)` backgrounds
  with `*-300`-level text (e.g. `#6ee7b7`), or keep pastel chips — verify 4.5:1 contrast.
- Shadows remain black-based (they read as depth on dark too); tinted `--shadow-primary`
  switches to the brighter green.

## 18. Responsive breakpoints

Tailwind defaults, three that matter:

```
sm: 640px   — stat cards 1→2 cols; timer number grows
md: 768px   — content gaps 12→16px; main padding 12→16px; text steps up (title 20→24px)
lg: 1024px  — sidebar appears (fixed, ml-64); stats →4 cols; main →3-col grid; padding 20px; title 30px
```

CSS: `@media (min-width: 640px|768px|1024px)`. AgentForce migrates its legacy `650px/1000px`
max-width queries to these min-width equivalents. Below `lg` the sidebar collapses to a hamburger
in a sticky top bar; controls stack to one column; grids collapse 4→2→1.

## 19. Animation / motion rules

Durations & easing: `duration-300` (interactions) and `duration-500 ease-out` (surfaces/entrances);
`transition-all` is standard on interactive elements.

Keyframes (from the reference, reproducible inline):

```css
@keyframes slideInUp   { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
@keyframes barSlideUp  { from { opacity:0; transform:scaleY(0); } to { opacity:1; transform:scaleY(1); } }  /* origin: bottom */
@keyframes fade-in     { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:translateY(0); } }
@keyframes float       { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-5px); } }
@keyframes shimmer     { from { transform:translateX(-100%); } to { transform:translateX(100%); } }  /* skeletons */
@keyframes pulse       { 50% { opacity:.5; } }  /* notification dots, live badges */
```

Rules:
- Page/card entrance: `slideInUp .5s ease-out both`, staggered `animation-delay: i*100ms`.
- Charts animate on load (`barSlideUp`, staggered per bar).
- Micro-interactions: scale 1.02–1.10 on hover only; decorative rotate ≤12°.
- Skeletons: `--muted` blocks + shimmer overlay.
- Always wrap in `@media (prefers-reduced-motion: reduce) { * { animation:none; transition:none; } }`.

## 20. Component composition rules

- **AppShell** = fixed Sidebar (§8) + sticky TopBar + `<main class="p-3 md:p-4 lg:p-5 lg:ml-64 space-y-3 md:space-y-4">`.
- **TopBar** = search field (§11, with ⌘F hint) · spacer · icon buttons with notification dots · avatar + name/email block (`text-sm font-semibold` / `text-xs text-muted-foreground`).
- **PageHeader** = `h1` + `text-sm text-muted-foreground` subtitle + action row (`flex gap-2 mt-4`: one primary + one outline button). Header block animates in first.
- **StatCard** = label (`text-xs font-medium opacity-90`) + top-right `w-6 h-6 rounded-full bg-primary-foreground/20` icon chip + `text-3xl font-bold` value + `text-xs opacity-80` trend row with 12px icon. Exactly **one** StatCard per row uses the primary variant.
- **Panel(Card) composition**: CardHeader (title + legend/action) → content → optional footer stats row separated by `border-t border-border pt-4`.
- **ListRow** (§13) composes Avatar/EmojiChip + two-line text + trailing StatusBadge.
- Icons: lucide-style 1.5–2px stroke, `16px` default (`[&_svg]:size-4`), 12px inside chips.
- Hierarchy recipe: neutral cards everywhere; **one** inverted (primary or foreground) card per
  view as the focal point; pastel chips for status; green accents for numbers that matter.
- Density: cards never nest deeper than 2 levels; content inside cards uses `gap-3/4/6`, never margins on both axes.

---

## Adaptation map for AgentForce (current → new)

| Current (`visualization.py`) | New token |
|---|---|
| `--bg #f4f6fa / #0d111b` | `--background #f8f9f5 / #040705` |
| `--panel #fff / #161d2b` | `--card #fff / #080c09` |
| `--text #172033 / #ecf1ff` | `--foreground #202318 / #f6f9f7` |
| `--muted #667085 / #a7b0c3` | `--muted-foreground #707367 / #86938a` |
| `--line #d9deea / #303a4e` | `--border #e3e6de / #1a251d` |
| `--accent #3157d5 / #8ea8ff` (blue) | `--primary #005e30 / #008b46` (green) |
| `--good / --warn` | status tokens (§1) |
| radius 7–12px | `--radius-sm/md/lg/xl` 12/14/16/20px |
| `box-shadow: 0 3px 14px #0001` | `--shadow-sm` rest / `--shadow-lg` hover |
| breakpoints 650/1000 (max-width) | 640/768/1024 (min-width) |

Constraints that must be preserved: self-contained single file, CSP-safe (inline style/script,
`img-src data:` only), Vietnamese labels, keyframe-gallery information architecture,
filter/sort/pagination behavior.
