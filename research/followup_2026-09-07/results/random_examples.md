# Random examples from the main N=100 evaluation

Ten of 141 eligible items, drawn uniformly without replacement with NumPy seed 20260907. Nothing is filtered for model or lens success. Ranks below are ordinary one-based ranks, minimized across each loop's 48 layers; they are descriptive oracle summaries. Trailing prompt whitespace is trimmed only in this display. Exact prompts, exits, scores and ranks are in `random_examples.json`.

## letterpos-nitrogen-symbol

```text
Fact: The position in the alphabet of the chemical symbol for nitrogen is
```

Task labels: ['N']; target: '14'; continuation: '14.\n'; historical pass: True.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[1106, 146, 11, 14]]; excess hit@10: [-0.014, -0.116, -0.101, -0.246].
- logitlens: best ranks by eligible slot, loops 1–4: [[6, 19, 13, 3]]; excess hit@10: [0.812, -0.203, -0.174, 0.739].

## chem-bones-Z

```text
Fact: The atomic number of the metal most abundant in human bones is
```

Task labels: ['calcium']; target: '20'; continuation: '24.\n'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[3, 24, 151, 1]]; excess hit@10: [0.957, -0.159, -0.087, 0.826].
- logitlens: best ranks by eligible slot, loops 1–4: [[24, 92, 1, 10]]; excess hit@10: [-0.217, -0.188, 0.783, 0.783].

## rhyme-door-doubled

```text
Fact: Take the number whose name rhymes with the word door. Doubling it gives the number
```

Task labels: ['four']; target: '8'; continuation: '12. What'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[51, 18, 12, 1]]; excess hit@10: [0.0, -0.087, -0.145, 0.783].
- logitlens: best ranks by eligible slot, loops 1–4: [[3, 2, 4, 4]]; excess hit@10: [0.841, 0.812, 0.768, 0.797].

## mult-parens-sub

```text
(9 - 6) * 2 =
```

Task labels: ['3', 'multiplication']; target: '6'; continuation: '6\n\n('; historical pass: True.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[16, 4, 5, 2]]; excess hit@10: [0.0, 0.769, 0.692, 0.538].
- logitlens: best ranks by eligible slot, loops 1–4: [[1, 1, 2, 2]]; excess hit@10: [0.385, 0.538, 0.538, 0.538].

## louvre-language

```text
Fact: The primary language spoken in the country where the Louvre museum is located is
```

Task labels: ['France']; target: 'French'; continuation: ' French.\n\n'; historical pass: True.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[85, 40, 5, 2]]; excess hit@10: [-0.029, 0.0, 1.0, 1.0].
- logitlens: best ranks by eligible slot, loops 1–4: [[15, 7, 6, 3]]; excess hit@10: [0.0, 1.0, 1.0, 1.0].

## rhyme-tree-squared

```text
Fact: The square of the number whose name rhymes with tree is
```

Task labels: ['three']; target: '9'; continuation: '121.'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[281, 10, 5, 3]]; excess hit@10: [-0.014, 0.899, 0.841, 0.812].
- logitlens: best ranks by eligible slot, loops 1–4: [[3, 2, 3, 3]]; excess hit@10: [0.826, 0.841, 0.841, 0.841].

## half-clock-hours

```text
Fact: Half the number of hours shown on a standard clock face is
```

Task labels: ['12']; target: '6'; continuation: '12 hours.'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[2594, 255, 41, 1]]; excess hit@10: [-0.029, -0.029, -0.101, 0.768].
- logitlens: best ranks by eligible slot, loops 1–4: [[52, 6, 1, 1]]; excess hit@10: [-0.188, 0.812, 0.768, 0.768].

## word-div-sub

```text
fifteen divided by three minus two equals
```

Task labels: ['5', 'subtraction']; target: 'three'; continuation: '\n\nfifteen'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[5167, 621, 30, 1]]; excess hit@10: [0.0, 0.0, -0.077, 0.308].
- logitlens: best ranks by eligible slot, loops 1–4: [[5, 7, 1, 1]]; excess hit@10: [0.231, 0.0, 0.538, 0.385].

## etym-janus-monthnum

```text
Fact: The month named after the two-faced Roman god of doorways is month number
```

Task labels: ['January']; target: '1'; continuation: '11.\n'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[1425, 276, 16, 3]]; excess hit@10: [-0.014, -0.159, -0.087, 0.754].
- logitlens: best ranks by eligible slot, loops 1–4: [[2, 3, 2, 4]]; excess hit@10: [0.768, 0.754, 0.754, 0.797].

## rhyme-hive-plusone

```text
Fact: One more than the number whose name rhymes with hive is
```

Task labels: ['five']; target: '6'; continuation: '100.'; historical pass: False.

- jlens_exit3: best ranks by eligible slot, loops 1–4: [[306, 25, 7, 4]]; excess hit@10: [-0.014, 0.0, 0.87, 0.812].
- logitlens: best ranks by eligible slot, loops 1–4: [[5, 5, 4, 7]]; excess hit@10: [0.826, 0.826, 0.797, 0.797].
