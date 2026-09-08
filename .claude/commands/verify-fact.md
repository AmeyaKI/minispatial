---
description: Re-check one line in context/FACTS.md at its source and update its tag and date
---

Takes a line from `context/FACTS.md` (quote it, or name the fact). Rule 11: anything tagged
`unverified` must be checked before it is used.

## 1. Identify the source

Not a source that would confirm it — **the** source. For a registry name, the installed package's
registry. For a config value, the vendored config file. For a published metric, the paper or model
card, at its URL. For a bucket path, the bucket.

If the fact has no locatable source, that is the finding: mark it `unsourced` and stop. Do not
promote it to `verified` because it seems right.

## 2. Check it

Prefer executing over reading:

```bash
# registry names
.venv/bin/python -c "from terratorch.registry import BACKBONE_REGISTRY; print([n for n in BACKBONE_REGISTRY if 'prithvi' in n])"

# installed versions
.venv/bin/python -c "from importlib.metadata import version; print(version('coremltools'))"

# config values
.venv/bin/python -c "import yaml; print(yaml.safe_load(open('train/configs/reference/sen1floods11.yaml'))['data']['init_args']['bands'])"
```

For an external document, fetch it and quote the sentence containing the value. Do not record a
number you did not read this session.

## 3. Update the line

Rewrite it in place with: the tag (`verified` / `unverified` / `unsourced`), the source (a URL, a
file path, or the exact command), and today's date. If the source disagrees with what was recorded,
**change the fact** and note the correction — a fact that turned out wrong is more valuable than
one that was never checked.

## 4. Propagate

If the fact changed, grep for anything that depended on it:

```bash
grep -rn "<the old value>" --include=*.md --include=*.py . | grep -v '.venv'
```

Anything built on the old value goes to `context/BLOCKERS.md`, and if the fact drove a decision,
add a correcting entry to `context/DECISIONS.md`. Do not edit an existing decision entry in place —
the record of having believed the wrong thing is part of the decision log.
