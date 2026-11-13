# Writing these docs

The site is built with MkDocs Material from `docs/site/`. Build it with `mkdocs build --strict` after `pip install -r docs/requirements.txt`. Two tests guard it and run with `make test`: `tests/test_docs_style.py` and `tests/test_docs_constants.py`.

## Values from the code

Never type a constant from the code into a page. Ask for it with a macro from `docs/macros.py`. The page then shows the current value, and the build fails when the name no longer exists:

| Macro | Gives | Example |
| --- | --- | --- |
| `const("module.NAME")` | a client module attribute | `{{ "{{" }} const("ombra.presets.FILTER_LAMBDA") {{ "}}" }}` gives {{ const("ombra.presets.FILTER_LAMBDA") }} |
| `preset("name", "key")` | a value of a product preset | `{{ "{{" }} preset("fast", "ring_dim") {{ "}}" }}` gives {{ preset("fast", "ring_dim") }} |
| `fmax("name")` | the filter field limit $F_{max}$ of a preset | `{{ "{{" }} fmax("balanced") {{ "}}" }}` gives {{ fmax("balanced") }} |
| `cpp("path", "kName")` | a `constexpr` from a C++ file | `{{ "{{" }} cpp("core/include/ombra/core/x.hpp", "kName") {{ "}}" }}` |

Macros work inside math too: `$\lambda = {{ "{{" }} const("ombra.presets.FILTER_LAMBDA") {{ "}}" }}$`.

Two opening braces in a row start a macro, so LaTeX like {% raw %}`\frac{{a}}{b}`{% endraw %} breaks the page. Put a space between them (`\frac{ {a} }{b}`) or wrap the formula in `{{ "{%" }} raw {{ "%}" }}` and `{{ "{%" }} endraw {{ "%}" }}`.

## Deep-dive sections

Mark a page or a section that a reader can skip with the deep-dive admonition:

```markdown
!!! deep-dive "Why the expansion uses no level"
    The details...
```

!!! deep-dive "What it looks like"
    A box like this one. Use `??? deep-dive "Title"` for one that starts folded.

## Style

The style test flags em and en dashes, a hyphen between spaces used as a dash, emoji, and these words and phrases: `seamlessly`, `robust`, `leverage`, `cutting-edge`, `delve`, `unlock`, `empower`, `it's worth noting`, `in today's world`, `not just ... but`, and `open source` (Ombra is source-available). It skips code blocks and inline code. Beyond what the test checks: plain words, short sentences, active voice, and numbers instead of adjectives. Say plainly what is not supported or slow. Each formula comes after a sentence that says what it does, and is followed by a small worked example.
