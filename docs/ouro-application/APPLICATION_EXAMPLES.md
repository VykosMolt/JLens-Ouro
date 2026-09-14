# Randomly selected examples

Five items drawn with `random.Random(0).sample` from the 142 items that have at least one scorable, non-leaked intermediate (of 148 stimuli). Not cherry-picked. For each loop: the model's own top-1 at that loop's exit, then the best rank of the true intermediate over the 48 layers (0 = top-1; hit@10 means rank below 10) under the final-exit Jacobian lens and under the logit lens, with the three names each lens ranked highest at that layer. Source: `artifacts/jlens/eval/round1_exit3x32/{items.json,arrays.npz}`.

## atomic-26-symbol (multihop)

- Prompt: `Fact: The chemical symbol for the element with atomic number 26 is`
- Target: `Fe`. Model continuation: `Fe.`. Correct: True.
- Exit top-1, loops 1 to 4: ['"', 'Fe', 'Fe', 'Fe']
- Intermediate `iron`:
  - loop 1: Jacobian lens rank 979 (layer 5; top-3 ['11', '14', '12']); logit lens rank 0 (layer 43; top-3 ['iron', 'copper', 'gold'])
  - loop 2: Jacobian lens rank 1716 (layer 3; top-3 ['gold', 'H', 'mercury']); logit lens rank 0 (layer 40; top-3 ['iron', 'copper', 'carbon'])
  - loop 3: Jacobian lens rank 222 (layer 43; top-3 ['N', 'H', 'C']); logit lens rank 0 (layer 40; top-3 ['iron', 'copper', 'carbon'])
  - loop 4: Jacobian lens rank 0 (layer 41; top-3 ['iron', 'copper', 'calcium']); logit lens rank 0 (layer 40; top-3 ['iron', 'copper', 'carbon'])

## rhyme-hive-plusone (multihop)

- Prompt: `Fact: One more than the number whose name rhymes with hive is`
- Target: `6`. Model continuation: `100.`. Correct: False.
- Exit top-1, loops 1 to 4: ['1', '1', '1', '1']
- Intermediate `five`:
  - loop 1: Jacobian lens rank 3 (layer 20; top-3 ['7', 'six', '6']); logit lens rank 4 (layer 45; top-3 ['3', 'three', 'third'])
  - loop 2: Jacobian lens rank 53 (layer 45; top-3 ['7', '13', '5']); logit lens rank 4 (layer 0; top-3 ['3', 'three', 'third'])
  - loop 3: Jacobian lens rank 7 (layer 13; top-3 ['7', '5', 'five']); logit lens rank 3 (layer 0; top-3 ['7', '5', 'five'])
  - loop 4: Jacobian lens rank 2 (layer 3; top-3 ['7', 'six', '6']); logit lens rank 6 (layer 45; top-3 ['7', 'six', '6'])

## add-sub-left-right (order-ops)

- Prompt: `10 + 5 - 3 =`
- Target: `12`. Model continuation: `12`. Correct: True.
- Exit top-1, loops 1 to 4: ['1', '1', '1', '1']
- Intermediate `15`:
  - loop 1: Jacobian lens rank 39 (layer 45; top-3 ['15', '7', '13']); logit lens rank 2 (layer 23; top-3 ['13', '15', '16'])
  - loop 2: Jacobian lens rank 1 (layer 17; top-3 ['15', '13', '11']); logit lens rank 9 (layer 24; top-3 ['13', '12', '16'])
  - loop 3: Jacobian lens rank 16 (layer 4; top-3 ['13', '11', '15']); logit lens rank 23 (layer 42; top-3 ['15', '12', 'squared'])
  - loop 4: Jacobian lens rank 3 (layer 17; top-3 ['11', '13', '15']); logit lens rank 4 (layer 41; top-3 ['15', '11', '12'])

## nested-add-mult-sub (order-ops)

- Prompt: `((1 + 2) * 3) - 4 =`
- Target: `5`. Model continuation: `9

((`. Correct: False.
- Exit top-1, loops 1 to 4: ['1', '9', '9', '9']
- Intermediate `9`:
  - loop 1: Jacobian lens rank 29 (layer 10; top-3 ['6', '8', '7']); logit lens rank 0 (layer 24; top-3 ['9', '3', '7'])
  - loop 2: Jacobian lens rank 17 (layer 11; top-3 ['15', '7', '5']); logit lens rank 0 (layer 1; top-3 ['9', '3', '6'])
  - loop 3: Jacobian lens rank 0 (layer 3; top-3 ['9', '7', '5']); logit lens rank 0 (layer 1; top-3 ['9', '7', '5'])
  - loop 4: Jacobian lens rank 0 (layer 0; top-3 ['9', '7', '5']); logit lens rank 0 (layer 42; top-3 ['9', '7', '5'])

## word-sub-mult (order-ops)

- Prompt: `ten minus two times three equals`
- Target: `four`. Model continuation: `Answer:`. Correct: False.
- Exit top-1, loops 1 to 4: ['', '', '', '']
- Intermediate `6`:
  - loop 1: Jacobian lens rank 179 (layer 29; top-3 ['squared', 'division', '6']); logit lens rank 8 (layer 44; top-3 ['9', '6', '12'])
  - loop 2: Jacobian lens rank 1725 (layer 35; top-3 ['subtraction', '3', '13']); logit lens rank 4 (layer 44; top-3 ['10', '9', '6'])
  - loop 3: Jacobian lens rank 213 (layer 35; top-3 ['3', 'subtraction', '4']); logit lens rank 6 (layer 44; top-3 ['10', '9', '7'])
  - loop 4: Jacobian lens rank 12 (layer 46; top-3 ['6', '10', '7']); logit lens rank 11 (layer 46; top-3 ['10', '3', '6'])

