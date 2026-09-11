# Primary parallel-upload review: rejected pending two focused fixes

The 12 supplied offline tests pass. Independent probes expose two blockers: cancellation can miss a subprocess registered immediately afterward, and the remote file hash can be computed from a different inode than the path whose metadata is checked. The second probe reported an existing bank complete while its actual SHA-256 was wrong. These observations and exact candidate source pins are retained in the JSON companions.

The helper needs cancellation-aware process registration and opened-descriptor identity checks, with the corresponding chunk and temporary-publication checks. The existing self-test should retain both regressions before renewed exact-hash acceptance. No real lease, scientific payload or provider resource was changed by this review.

The retry accounting and administrative scope pass review. Attempt01 is terminated with three matching absence confirmations and a $0.2258840927 charge bound. Only setup changes, from 3,600 to 12,300 seconds; the original freeze, specification, bundle and controller remain unchanged. The new full deadline envelope costs at most $3.7410616438 under the fixed rate and latency margin, and the retry cap is $3.77. First attempt plus retry cap is $3.9958840927, below the aggregate $4 primary ceiling. No live deadline was extended, and no new primary model outcome was inspected.

This review does not accept use of the current helper. The original launcher, bootstrap, development receipt and scientific gates remain required after a corrected helper pre-stages the original file bytes and mtimes.
