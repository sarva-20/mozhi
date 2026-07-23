# Mozhi Configuration

This directory contains all domain-specific configuration for Mozhi.
No Python knowledge required — edit these JSON files to adapt Mozhi
to your domain.

---

## ontology.json

Add domain-specific terms Mozhi should recognize and correct to.

```json
{
  "terms": [
    "your domain term",
    "multi word term",
    "another term"
  ],
  "weights": {
    "phonetic": 0.5,
    "jaro": 0.3,
    "token": 0.2
  }
}
```

`terms` — list of words or phrases in your domain.
`weights` — scorer weight overrides. Must sum to 1.0. Remove this
block entirely to use Mozhi defaults.

---

## acronyms.json

Map how your acronyms sound when spoken aloud to their correct form.

```json
{
  "enunciated form": "ACRONYM",
  "eye pee see": "IPC",
  "eye bee em": "IBM"
}
```

Keys are lowercase enunciated forms. Values are the correct acronym.

---

## stopwords.json

Add or remove stopwords from Mozhi's base English set.

```json
{
  "add": ["hereby", "whereas"],
  "remove": []
}
```

`add` — domain filler words that should never be matched against
your ontology.
`remove` — words in the base English stopword list that are
meaningful in your domain and should not be skipped.