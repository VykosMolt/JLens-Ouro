# Real benchmark examples (selection rule stated; no example chosen by outcome)

Selection: the three fixed numerical-sample items of the frozen scorer (`readouts.SAMPLE_ITEMS = (0, 1, 2)` = confirmation-001..003), the first item in ID order with a zero-token boundary drop, and the W-collision item confirmation-027. Token IDs are the exact model input (`token_ids`); the first ID is BOS. Readout position is the last token.

## confirmation-001 (astronomy, dg001)

- Prompt: `Fact: The planet whose sidereal rotation takes longer than its orbital period has an atmosphere dominated by `
- Target (answer, never shown to the model): `carbon dioxide`
- Annotated intermediate: **Venus**; scored single-token forms (ID → text): 16293 → ` Venus`
- Model input (20 tokens incl. BOS; boundary drop 1): IDs `[0, 13486, 42, 378, 3925, 3449, 2103, 6918, 10792, 2933, 2848, 670, 624, 19865, 1657, 553, 354, 5264, 12742, 411]`
- Decoded tokens: '<|endoftext|>' | 'Fact' | ':' | ' The' | ' planet' | ' whose' | ' side' | 'real' | ' rotation' | ' takes' | ' longer' | ' than' | ' its' | ' orbital' | ' period' | ' has' | ' an' | ' atmosphere' | ' dominated' | ' by'
- Readout token (position −1): ` by`; controls: 79 other names; relation: atmospheric_composition (familiar_fact_relation_recombined)

## confirmation-002 (astronomy, dg001)

- Prompt: `Fact: The planet with persistent sulfuric-acid clouds rotates in the direction described as `
- Target (answer, never shown to the model): `retrograde`
- Annotated intermediate: **Venus**; scored single-token forms (ID → text): 16293 → ` Venus`
- Model input (18 tokens incl. BOS; boundary drop 1): IDs `[0, 13486, 42, 378, 3925, 351, 10499, 17255, 286, 29, 24529, 10695, 35290, 281, 260, 4376, 3873, 347]`
- Decoded tokens: '<|endoftext|>' | 'Fact' | ':' | ' The' | ' planet' | ' with' | ' persistent' | ' sulfur' | 'ic' | '-' | 'acid' | ' clouds' | ' rotates' | ' in' | ' the' | ' direction' | ' described' | ' as'
- Readout token (position −1): ` as`; controls: 79 other names; relation: rotation_direction (not_represented_requested_relation)

## confirmation-003 (astronomy, dg002)

- Prompt: `Fact: The planet whose rotation axis is tilted nearly into its orbital plane had its rings discovered in `
- Target (answer, never shown to the model): `1977`
- Annotated intermediate: **Uranus**; scored single-token forms (ID → text): 36985 → ` Uranus`
- Model input (21 tokens incl. BOS; boundary drop 0): IDs `[0, 13486, 42, 378, 3925, 3449, 10792, 6867, 314, 37929, 3920, 618, 624, 19865, 8303, 761, 624, 13413, 3658, 281, 216]`
- Decoded tokens: '<|endoftext|>' | 'Fact' | ':' | ' The' | ' planet' | ' whose' | ' rotation' | ' axis' | ' is' | ' tilted' | ' nearly' | ' into' | ' its' | ' orbital' | ' plane' | ' had' | ' its' | ' rings' | ' discovered' | ' in' | ' '
- Readout token (position −1): ` `; controls: 79 other names; relation: ring_discovery_year (not_represented_requested_relation)

## confirmation-003 (astronomy, dg002)

- Prompt: `Fact: The planet whose rotation axis is tilted nearly into its orbital plane had its rings discovered in `
- Target (answer, never shown to the model): `1977`
- Annotated intermediate: **Uranus**; scored single-token forms (ID → text): 36985 → ` Uranus`
- Model input (21 tokens incl. BOS; boundary drop 0): IDs `[0, 13486, 42, 378, 3925, 3449, 10792, 6867, 314, 37929, 3920, 618, 624, 19865, 8303, 761, 624, 13413, 3658, 281, 216]`
- Decoded tokens: '<|endoftext|>' | 'Fact' | ':' | ' The' | ' planet' | ' whose' | ' rotation' | ' axis' | ' is' | ' tilted' | ' nearly' | ' into' | ' its' | ' orbital' | ' plane' | ' had' | ' its' | ' rings' | ' discovered' | ' in' | ' '
- Readout token (position −1): ` `; controls: 79 other names; relation: ring_discovery_year (not_represented_requested_relation)

## confirmation-027 (chemistry, dg006)

- Prompt: `Fact: The metal with the highest melting point has the chemical symbol `
- Target (answer, never shown to the model): `W`
- Annotated intermediate: **tungsten**; scored single-token forms (ID → text): 41387 → ` tungsten`
- Model input (14 tokens incl. BOS; boundary drop 1): IDs `[0, 13486, 42, 378, 4718, 351, 260, 4919, 14004, 1225, 553, 260, 2819, 3573]`
- Decoded tokens: '<|endoftext|>' | 'Fact' | ':' | ' The' | ' metal' | ' with' | ' the' | ' highest' | ' melting' | ' point' | ' has' | ' the' | ' chemical' | ' symbol'
- Readout token (position −1): ` symbol`; controls: 79 other names; relation: chemical_symbol (familiar_requested_family)
