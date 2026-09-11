# Independent parallel input-transfer review

Status: **passed** for `resources/parallel_upload.py`, 35,465 bytes, SHA-256 `512946233e308a7918f1871eff88beca049afcf733d2382a0efc12cbab8ec2f4`.

The reviewed helper preserves the four original bank hashes and invokes the unchanged launcher after verified pre-staging. It binds the controller, lease identity and fixed deadlines, limits transport to eight streams, verifies chunk and whole-bank content plus opened-descriptor identity, and publishes durable files with the source mtime.

All 16 supplied self-tests passed in an independent local rerun. Four additional independent regressions passed: one cancellation before child registration, the original parent-directory swap during remote open, one cancellation of a SIGTERM-ignoring child, and one cancellation of a SIGTERM-ignoring parent and descendant. Each process fixture completed with its process group quiescent. No cloud, SSH or new confirmation results were accessed.

Three earlier findings are resolved. The registration race now kills late children under the cancellation lock. Remote hashes and assembly bind opened descriptors to the pathname and recheck the completed temporary file before publication. Cancellation sends SIGKILL to owned groups immediately; original transport cleanup verifies no live members remain. Rejected v1 review and proof bytes are preserved separately; the intermediate SIGTERM finding is recorded under the v2 additional-rejection artifact.

The prospective retry configuration changes only setup time from 3,600 to 12,300 seconds. Its declared full envelope is $3.741061643835616. Including the first terminated attempt ($0.22588409273346807), the primary envelope is $3.9669457365690843; the first charge plus the $3.77 retry job cap is $3.9958840927334682, within the $4 primary ceiling. These checks do not extend a live lease.

This acceptance covers the pinned transfer helper and prospective retry calculation. Controller admission, the independent watcher, native development acceptance, the frozen scientific worker, receiving validation and termination remain required. See the accompanying JSON and proof for exact records and execution evidence.
