# Blinded dependency and relation audit

The 160 revised candidates form **28 connected dependency groups**, not 80 independent concept groups. Their item-weighted effective group count is **8.625337**. No new confirmation outcomes were inspected.

Candidate SHA-256: `ac2e54822d10d89743a9857d0225377db1c161ccf6ea686c9d36550001697a75`. The machine-readable companion contains all 294 edges with reasons, complete components, and all 160 requested-relation annotations.

## Grouping decisions

A concrete construction keeps the intermediate entity type and requested slot, or a closely corresponding typed clue/request pair. Named-entity substitutions and paraphrases within those constructions stay connected. The Fact prefix, generic clue->entity->property schema, broad domain, source publisher/page, and isolated broad relation taxonomy are not sufficient alone. Examples: SI unit->symbol is one family; element->chemical symbol is another, joined here by the required W answer collision. Part counts across biological subunits and network encodings are a broad familiar relation, not by that fact alone one concrete entity-substitution construction.

Casefold and collapse whitespace; manually inspect semantics. W/W merges even though tungsten and watt use the same surface answer for different referents. Water versus ice are different state-specific answers; krone versus krona are different national currency names. All candidate target_aliases are empty.

Reviewed both identity clues and requested facts. Explicit shared propositions are listed as edge reasons. No additional cross-concept shared proposition beyond listed identity/template/answer connections was identified; sharing a broad source page or a mentioned spacecraft does not itself equate the facts.

| Group | Size | Items (numeric suffix) | Intermediates |
|---|---:|---|---|
| dg001 | 2 | 1, 2 | Venus |
| dg002 | 4 | 3, 4, 117, 118 | Uranus, Voyager |
| dg003 | 10 | 5, 6, 11, 12, 13, 14, 15, 16, 159, 160 | Neptune, Titan, Europa, Io, permafrost |
| dg004 | 2 | 7, 8 | Pluto |
| dg005 | 2 | 9, 10 | Ceres |
| dg006 | 42 | 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 53, 54, 59, 60, 63, 64, 75, 76, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146 | helium, lithium, boron, cobalt, nickel, tungsten, radon, neon, Andes, Alps, Borneo, nucleus, Kelvin, Pascal, Newton, Watt, Tesla, volt, mole, gray, ozone |
| dg007 | 16 | 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48 | Japan, Kenya, Chile, Norway, Egypt, Peru, Thailand, Sweden |
| dg008 | 4 | 49, 50, 93, 94 | Everest, Confucius |
| dg009 | 2 | 51, 52 | Sahara |
| dg010 | 6 | 55, 56, 57, 58, 61, 62 | Nile, Danube, Mekong |
| dg011 | 4 | 65, 66, 67, 68 | DNA, RNA |
| dg012 | 2 | 69, 70 | insulin |
| dg013 | 10 | 71, 72, 77, 78, 113, 114, 115, 116, 123, 124 | hemoglobin, chlorophyll, Linux, Python, Unicode |
| dg014 | 2 | 73, 74 | neuron |
| dg015 | 2 | 79, 80 | collagen |
| dg016 | 24 | 81, 82, 83, 84, 85, 86, 87, 88, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 111, 112 | Napoleon, Darwin, Einstein, Galileo, Shakespeare, Beethoven, Mozart, Picasso, Monet, Tolkien, Bach, Raphael |
| dg017 | 2 | 89, 90 | Cleopatra |
| dg018 | 2 | 91, 92 | Gutenberg |
| dg019 | 2 | 109, 110 | Chopin |
| dg020 | 2 | 119, 120 | Hubble |
| dg021 | 2 | 121, 122 | Wikipedia |
| dg022 | 2 | 125, 126 | Ethernet |
| dg023 | 2 | 127, 128 | Arduino |
| dg024 | 4 | 147, 148, 149, 150 | basalt, granite |
| dg025 | 2 | 151, 152 | monsoon |
| dg026 | 2 | 153, 154 | glacier |
| dg027 | 2 | 155, 156 | tornado |
| dg028 | 2 | 157, 158 | earthquake |

## Precision before eligibility filtering

Group sizes: `[42, 24, 16, 10, 10, 6, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`. Sum of squared sizes is 2968; effective groups are `160² / 2968 = 8.625337`.

| Assumed SD of paired group means | Approximate 95% half-width |
|---:|---:|
| 0.30 | 0.200212 |
| 0.45 | 0.300317 |
| 0.60 | 0.400423 |

Planning approximation with common group-mean variance and size weights; not an observed SD, interval, effect, power guarantee, or substitute for the prespecified group bootstrap. Large unequal groups dominate information. Recompute graph and precision after every eligible-item exclusion; do not merely delete empty concept IDs. The nominal 80-group precision claim is not supported by this candidate graph.

## Relation exposure versus discovery

Typed requested predicate; familiar_requested_family also includes the stated broad geographic/orbit/count/season relation when the entity class changes. familiar_fact_relation_recombined flags known component/location/release/etymology relations used in another hop or with a changed requested slot. not_represented_requested_relation means not found at this granularity in these 93 prompts only. These are annotations, not 160 independent relation families.

| Status | Item count |
|---|---:|
| familiar_fact_relation_recombined | 19 |
| not_represented_requested_relation | 96 |
| familiar_requested_family | 45 |

The first 80 versus last 80 split does not define familiarity. Retain the three statuses, or disclose any binary collapse. Do not call every unfamiliar requested predicate a wholly new reasoning operation.

The companion JSON gives every item its actual requested predicate, exposure status, discovery examples, and concrete template membership. Familiar requested relations occur in both halves of the draft; many first-half predicates are absent from discovery. The original relation_type fields should not be used to define the mixture.

## Concrete template edges

| Template family | Items | Construction |
|---|---|---|
| element_atomic_number | 17, 19, 22, 24, 26, 28, 30 | Descriptive element clue -> element -> atomic number; the chemical element/clue is substituted. |
| element_chemical_symbol | 20, 23, 25, 27, 29, 31 | Element clue -> element -> chemical symbol; includes atomic-number and functional clues. |
| country_currency | 33, 35, 37, 39, 41, 43, 45, 47 | Country clue -> country -> currency name; landmark, city, geographic, or flag clue is substituted. |
| country_capital | 34, 38, 42, 44, 46, 48 | Country clue -> country -> capital city; country/clue is substituted. |
| person_birthplace | 82, 84, 85, 87, 95, 97, 99, 101, 106, 107, 111 | Biographical achievement or authored-work clue -> person -> birthplace; city/town/island slot varies. |
| work_creator_birth_year | 96, 100 | Authored-work clue -> creator -> birth year; literary/music entity substitution. |
| painting_creator_movement | 102, 104 | Named painting -> painter -> associated/developed artistic movement. |
| si_unit_symbol | 129, 132, 133, 136, 137, 139, 141, 143 | Definitional/eponym clue -> SI unit -> official unit symbol. |
| si_unit_quantity | 130, 131, 134, 135, 138, 140, 142 | Definitional/eponym clue -> SI unit -> measured physical quantity. |
| mountain_range_continent | 53, 59 | Range identified by its named highest summit -> range -> continent. |
| geographic_entity_highest_summit | 54, 60, 64 | Geographic region identified by location/landmark -> region -> highest summit. |
| river_receiving_water_body | 55, 57 | River identified by branches/cities -> river -> receiving sea. |
| city_pair_river_terminus | 57, 61 | River identified by two named cities -> river -> mouth/delta destination. |
| river_headwaters | 58, 62 | River identified by a distinctive downstream location -> river -> headwaters location. |
| nucleic_acid_sugar | 65, 67 | Nucleic acid identified by characteristic structure/base -> nucleic acid -> sugar component. |
| functional_biomolecule_metal | 71, 77 | Functionally identified biological molecule -> molecule -> contained/central metal. |
| ranked_moon_liquid_reservoir | 11, 13 | Moon identified by size rank and parent system -> moon -> principal liquid in surface/subsurface reservoir. |
| spacecraft_observed_moon_parent | 12, 16 | Moon identified by a dated spacecraft observation/landing -> moon -> parent planet. |
| landmark_rock_igneous_class | 147, 149 | Named landmark -> characteristic rock -> intrusive/extrusive igneous class. |
| software_initial_release | 114, 115 | Software identified by mascot/creator -> software -> initial public release year. |

Every pair within each listed template family is linked. All 80 same-intermediate pairs are linked independently. The JSON also lists the explicit shared-fact reasons within those pairs.

Final-answer collisions: `1977`: 3, 118; `1991`: 114, 115, 123; `2`: 17, 76; `3`: 19, 63, 146; `4`: 72, 124; `china`: 50, 93; `methane`: 6, 11, 160; `w`: 27, 136. The final assembled alias metadata also gives `W` for tungsten and Watt. This is a shared surface code for distinct concepts, not semantic identity. It requires a collision edge; those concepts are already connected by their `W` final-answer collision, so component membership is unchanged. The original audit JSON records its earlier source snapshot; the final integration review records this metadata update.

## Required before freeze

Use the audited connected components and recomputed precision. Replace stale relation annotations, and bind independently checked hop claims to the final question text. This audit does not certify external factual sources or token eligibility. Any prompt/answer revision changes the source hash and requires checking affected edges; any eligibility exclusion requires recomputing components.

The bootstrap is conditional on this fit and this curated domain/relation/template mixture. It does not cover the variability from selecting entirely different template families, facts, models, or calibration fits. Large groups and the small effective group count materially limit precision.
