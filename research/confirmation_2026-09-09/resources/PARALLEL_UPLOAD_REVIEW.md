# Bounded parallel bank upload

`parallel_upload.py` is ready for independent review. It has performed local self-tests only: no live SSH, provider calls, worker creation or scientific evaluation.

Helper: 35465 bytes, SHA-256 `512946233e308a7918f1871eff88beca049afcf733d2382a0efc12cbab8ec2f4`.

## Mechanism and scope

The helper accepts only an exact canonical primary `--lease` root. It directly compiles verified current primary controller bytes, checks the lease's handoff binding, local sources, contract, SSH endpoint/identity and fixed deadlines, and reads the four unique frozen bank records from that bound run specification. It pins the unchanged `evaluation/launch.py` and its `artifacts.py` dependency. No scientific source, bank, configuration, receipt, controller limit or original launcher is modified.

Production settings are fixed at **eight concurrent workers and 64 MiB chunks**; there is no CLI override. The four bank destinations are exactly `banks/fit01.pt`, `banks/fit02.pt`, `banks/penultimate.pt`, and `banks/positions.pt`. Their total is 8,002,994,696 bytes, below the fixed 8 GiB chunk-cache limit.

Each original bank is hashed against its frozen record, split into new `resources/input_chunks/<bank-sha>/<offset>.part` files, and checked with before/after device/inode/size/mtime_ns/ctime_ns guards plus a full source SHA accumulated during splitting. Every chunk has its own SHA-256 in an immutable manifest. Existing cached chunks and manifests must agree; a changed canonical source fails. Chunk materialization checks local STOP/admission/deadline state as it proceeds.

For each bank, at most eight bounded SSH/rsync operations run concurrently through the controller's exact SSH options and host. Each chunk is locally hash-verified and its upload is preceded by fresh controller-source checks, unchanged local lease/binding/SSH/deadline checks and an owned remote `provider_lease_name`/STOP/deadline guard. Destinations are restricted to `/workspace/jlens/input_chunks/<bank-sha>/`. A fixed **300-second setup reserve** must remain; operation timeouts end before that reserve, and no deadline is extended.

Banks are processed sequentially. Remote assembly reads each chunk through the declared order and size, checks every chunk SHA and the complete original bank SHA, then fsyncs a temporary file inside `/workspace/jlens/banks`, sets the original bank's nanosecond mtime, atomically publishes the canonical bank, and fsyncs its directory. Opened descriptors are checked against the path identity before and after hashing/chunk reads; the publication temporary is reread and bound to its open descriptor through publication. A different existing canonical bank fails without overwrite. A matching existing canonical bank can resume safely. Only the declared temporary chunk files are removed after durable verified publication; manifests and assembly receipts remain.

Sequential assembly bounds temporary remote chunks plus the largest assembly copy to about 6.41 GB, plus bounded rsync partials, under roughly 8 GiB. Already published canonical banks are the intended final inputs and are separate from temporary storage. The local cache occupies about 8.003 GB plus at most one chunk's materialization temporary. A per-lease local lock and per-bank remote assembly lock prevent concurrent writers.

SIGINT/SIGTERM, local controller halt/STOP/admission change, a chunk error or the fixed setup reserve stops scheduling and terminates the uploader's started process groups. `transport.run_bounded` performs final process-group cleanup. Its records container synchronizes cancellation with registration and immediately kills any child registered after cancellation. Cancellation sends SIGKILL to its own started groups, so a child ignoring SIGTERM cannot delay cleanup until the transport timeout. Remote guards check `/workspace/jlens/STOP` and the fixed setup reserve before every upload and repeatedly while hashing/assembling. The helper does not perform provider lifecycle actions.

After all four canonical remote banks have verified original bytes/SHA and original mtimes, the helper rechecks originals and launcher sources, then invokes **the unchanged** `evaluation/launch.py --phase start`. Its original rsync `-t` checks can skip the identical bank files. Bootstrap, model/source/native smoke verification, BANKS_READY publication, external development acceptance and science ACK logic remain in their original code. The helper does not issue a development ACK and cannot authorize new science itself.

## Records and local validation

Each admitted invocation writes source/binding/record metadata and append-only fsynced events under `resources/parallel_upload_runs/<lease-name>/`. Events retain manifests, chunk completions, errors, assembly receipts and subprocess cleanup records. Chunk manifests remain under the cache; remote chunk manifests and assembly receipts remain on the worker.

The `--self-test` mode uses tiny local fixtures and executes the actual generated remote assembly program in a local temporary directory. **16 cases passed**, covering correct reassembly/SHA/mtime/resume, corrupt chunk, wrong full-bank SHA, wrong SSH owner marker, deadline reserve, remote STOP, immutable changed canonical bank, canonical source change during splitting, local identity/halt/deadline rejection, cancellation of an actual long-running local process group, a single cancellation before child registration, parent-directory swaps during both canonical-bank and chunk opens, and prompt cleanup of a ready-marked child that ignores SIGTERM. No separate test file was added.

Proof: `/tmp/parallel-upload-self-test-o3_pgqaa/PROOF.json`; its source record matches the helper bytes above. This local evidence does not establish live throughput or successful remote deployment.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/resources/parallel_upload.py --self-test
```

After the root agent admits the separately reviewed retry lease:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/moloch/ouro_project/venv/bin/python research/confirmation_2026-09-09/resources/parallel_upload.py --lease /absolute/path/to/admitted/primary/lease
```

The helper's controller pins must match that lease. Any controller source revision requires a new reviewed helper pin and rerun of the local self-test before use. Its transfer plan does not change the retry's aggregate spending authorization or native/science gates.
