# Independent precision correction implementation review

Passed for the candidate implementation. The original bundle matches every file in its parent freeze; the candidate has the same 57 files and differs only in evaluation/worker.py. Its complete AST is identical after removing the two redundant TF32 assignments; the remaining textual change is the explanatory comment. All fourteen evaluation/analysis sources are pinned in the JSON review. All scientific inputs and other sources remain byte-identical.

A fresh process with the pinned Torch build and frozen environment reproduces exact historical precision with defaults and CUDA uninitialized. The original setters change only matmul.fp32_precision from none to ieee. Executing the actual corrected comparison gate accepts matching defaults and rejects the setter-modified record. No model or scientific output was accessed. Remote precision still must pass its full gate.

The unchanged parallel helper must run from its original location, using the new lease. The archive must preserve bundle/ layout. The new specification hash and run ID must propagate into contract/config without adding unsupported schema keys. All controller/launcher/helper pins and bank records stay unchanged. A fresh publication namespace is required.

This review precedes final metadata construction. It does not approve an archive or receipt not yet produced; independently verify the final closure, parent provenance, new review binding, and corrected identity before launch.
