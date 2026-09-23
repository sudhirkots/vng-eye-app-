# Where the sample videos live

The nystagmus clips are **not in this repository** and never should be. They are
patient video, and this repo is public.

**Backed-up copy (the one that matters):**

```
C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG-EYE app videos\
```

28 files, ~369 MB. That folder sits in OneDrive, so it is cloud-backed, and it is
deliberately **outside** any git repository — nothing here can ever pick it up.

**Working copy:** `samples/` in this folder holds the same 28 files so the tools can
run without reaching across to OneDrive. It is gitignored. Treat it as disposable —
if it is ever lost, copy it back from the OneDrive folder above.

## Rules

- `samples/`, `*.mpg`, `*.MPG`, `*.mp4`, `*.avi` and `*.mov` are all gitignored.
  Check `.gitignore` before adding any new clip format.
- New clips go to the OneDrive folder **first**, then get copied into `samples/`.
  That way the backed-up copy is never the one that only exists on this laptop.
- Never `git add -f` anything under `samples/`.

## Copying between the two

Use robocopy, not `Copy-Item -Recurse` — this project has pnpm junctions under
`node_modules` that make Copy-Item follow the links and explode the tree.

```
robocopy "C:\Users\sudhi\OneDrive\Documents\AI apps made by me\VNG-EYE app videos" "C:\dev\vng-eye-app\samples" /E /XJ
```
