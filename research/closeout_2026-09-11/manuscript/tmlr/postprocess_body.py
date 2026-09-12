#!/usr/bin/env python3
"""Post-process the pandoc body for TMLR: drop Appendix D (the statement is a first-page footnote),
fix the Organization paragraph, and for the anonymous variant point the availability sentence at the
supplementary material instead of the repository.  usage: postprocess_body.py IN OUT submission|preprint"""
import re, sys
src, dst, variant = sys.argv[1:4]
s = open(src, encoding='utf-8').read()
s, n = re.subn(r'\\section\{Contribution and AI-assistance statement\}\\label\{app:D\}.*\Z', '', s, flags=re.S); assert n == 1, 'Appendix D not found'
s, n = re.subn(r';\s+Appendix D the\s+contribution\s+statement', '', s); assert n == 1, 'Organization sentence not found'
if variant == 'submission':
    sentence = 'published in the repository \\url{https://github.com/VykosMolt/JLens-Ouro} (research directory; files of 50 MB or more, including the banks and state caches, are listed there with hashes but not stored)'
    pattern = r'\s+'.join(re.escape(w) for w in sentence.split(' '))   # pandoc re-wraps lines; match across any whitespace
    s, n = re.subn(pattern,
                   'included in the supplementary material (files of 50 MB or more, including the banks and state caches, are listed there with hashes but not stored)', s)
    assert n == 1, 'availability sentence not found'
    assert 'github' not in s.lower(), 'repository reference left in the anonymous body'
assert 'Appendix D' not in s
open(dst, 'w', encoding='utf-8').write(s.rstrip('\n') + '\n')
