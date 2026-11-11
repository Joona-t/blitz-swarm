# SQLite WAL Mode Internals: Structure, Checkpointing, and Concurrency

## Executive Summary

SQLite's Write-Ahead Logging (WAL) mode, introduced in version 3.7.0 (2012), replaces the default rollback journal mode with an append-only log to improve concurrency and reduce fsync latency. This report details the binary structure of the WAL file, the precise conditions triggering checkpointing (passive, full, and truncate), and the rigorous concurrency guarantees provided, including the handling of readers during checkpoints and the specific consistency models applied.

## 1. WAL File Structure and Lifecycle

### 1.1 The Three-File System
When a database is opened in WAL mode (`PRAGMA journal_mode=WAL`), SQLite manages three files in the database directory:
1.  **Main Database File** (`db.sqlite`): The primary data store.
2.  **WAL File** (`db.sqlite-wal`): The write-ahead log. All modifications are appended here before being applied to the main database.
3.  **Shared Memory File** (`db.sqlite-shm`): A memory-mapped file containing a header, mutexes, and frames information. This is critical for multi-process synchronization. It is created automatically and removed when the last connection to the database is closed.

*Note: The `-shm` and `-wal` files are removed when the last connection closes, provided no other processes have the database open. If processes crash, these files may persist and be recovered on the next open.*

### 1.2 WAL File Header (32 Bytes)
The first 32 bytes of the WAL file constitute the WAL header. It is written when the WAL file is first created.

| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0 | 4 | Magic Number | `0x377f0682` (little-endian) or `0x377f0683` (big-endian). |
| 4 | 4 | File Format Version | Currently `3007000`. |
| 8 | 4 | Page Size | Page size in bytes. |
| 12 | 4 | Checkpoint Sequence Number | Incremented after each checkpoint. |
| 16 | 4 | Salt-1 | Random 32-bit number. Used to detect stale WAL files. |
| 20 | 4 | Salt-2 | Random 32-bit number. |
| 24 | 4 | Checksum-1 | First 32 bits of checksum of the first 24 bytes. |
| 28 | 4 | Checksum-2 | Second 32 bits of checksum of the first 24 bytes. |

*Source: SQLite Documentation, "WAL Format Specification" (last verified 2023).*

### 1.3 Frame Header (24 Bytes)
Each "frame" in the WAL contains one database page plus a 24-byte frame header.

| Offset | Size | Field | Description |
| :--- | :--- | :--- | :--- |
| 0 | 4 | Page Number | 1-based page number in the main database. |
| 4 | 4 | For Commit Records, The Size Of The Database File In Pages After The Commit. 0 Otherwise. | If non-zero, this frame is a commit record. The value is the new database size. |
| 8 | 4 | Salt-1 | Must match Salt-1 in the WAL header. |
| 12 | 4 | Salt-2 | Must match Salt-2 in the WAL header. |
| 16 | 4 | Checksum-1 | Cumulative checksum. First 32 bits of checksum of the first 20 bytes of the frame header plus the frame content, plus Checksum-2 of the previous frame. |
| 20 | 4 | Checksum-2 | Cumulative checksum. Second 32 bits of the same calculation. |

**Checksum Algorithm:**
The checksum is a simple two-32-bit-word addition. It uses native byte order for the checksum calculation but assumes the values are treated as 32-bit integers. This is intentionally not cryptographic but is designed to catch disk corruption or truncated writes.

*Methodological Note: The checksum is cumulative. Frame N's checksum depends on Frame N-1. This means a single corrupted frame invalidates all subsequent frames in the WAL from a reader's perspective.*

### 1.4 Frame Content and Validity
- A frame is **valid** only if its Salt-1 and Salt-2 match the WAL header, AND its checksum is valid (i.e., matches the computed cumulative checksum).
- If a frame fails validation, the reader assumes the WAL is truncated at that point. All subsequent frames are ignored.
- The **database size** is determined by the last *commit* frame (where the "Size" field is non-zero). If no commit frame is found, the database size is considered to be the size it was before the WAL began.

## 2. Checkpointing Mechanisms

Checkpointing is the process of copying frames from the WAL back into the main database file. This reclaims space in the WAL and updates the main database to reflect committed transactions.

### 2.1 Types of Checkpoints
SQLite defines three checkpoint modes, accessible via `PRAGMA wal_checkpoint` or internally:

#### 2.1.1 PASSIVE Checkpoint (Default)
- **Behavior:** Copies as many frames from the WAL to the main database as possible *without blocking readers*.
- **Blocking:** It will not overwrite pages in the main database that are currently being read by an active reader. If a reader is holding a snapshot that includes a page, the passive checkpoint stops before that page.
- **Result:** May result in an incomplete checkpoint. The WAL file is not truncated. The `-wal` file persists with the unreadable frames at the end (or in the middle, depending on layout).
- **Use Case:** Safe for high-concurrency environments where latency spikes are unacceptable.

#### 2.1.2 FULL Checkpoint
- **Behavior:** Attempts to copy *all* frames to the main database.
- **Blocking:** It will wait for active readers to finish before overwriting pages. If readers are active, the writer is blocked until the last reader closes its transaction.
- **Result:** Once successful, the WAL is truncated to zero length (the file is deleted or truncated, and the next write starts a new WAL file). The main database is fully up-to-date.
- **Use Case:** Maintenance windows or when disk space for the WAL needs to be reclaimed.

#### 2.1.3 TRUNCATE Checkpoint
- **Behavior:** Equivalent to a FULL checkpoint, but additionally truncates the WAL file to zero length.
- **Difference from FULL:** In SQLite, FULL and TRUNCATE often behave identically in modern versions regarding file deletion, but TRUNCATE explicitly ensures the file size is 0. The distinction is subtle: FULL guarantees the data is in the main DB; TRUNCATE guarantees the WAL file is empty. In practice, after a FULL checkpoint, the WAL is reused, not deleted, until a TRUNCATE is explicitly requested or the file is closed.
- **Clarification:** According to the SQLite source, a FULL checkpoint resets the WAL file size to 0 if no readers are active. A TRUNCATE checkpoint does the same but forces the reset even if the logic for FULL might leave the file open (though effectively the result is the same: WAL size 0).

*Source: SQLite `pragma.c` and `wal.c` implementation details (2023).*

### 2.2 Automatic Checkpointing
SQLite automatically triggers a checkpoint when the WAL file exceeds a certain size.
- **Threshold:** The default threshold is **1000 pages** (approximately 4MB for a 4KB page size).
- **Mechanism:** When a transaction commits and the WAL size exceeds this threshold, the thread that committed the transaction attempts a **PASSIVE** checkpoint.
- **Tunables:**
    - `PRAGMA wal_autocheckpoint=N`: Sets the threshold to N pages (default 1000).
    - `PRAGMA journal_mode=WAL`: Implies automatic passive checkpointing is enabled.

### 2.3 Checkpoint Sequence Number and Recovery
- Each checkpoint increments the **Checkpoint Sequence Number** in the WAL header.
- On startup, SQLite reads the WAL header. If the WAL file exists and the checksums are valid, it replays the WAL into the main database.
- If the WAL file is stale (e.g., from a previous session with a different Salt), it is ignored and deleted.

## 3. Concurrency Guarantees and Models

### 3.1 Reader-Writer Concurrency
WAL mode provides **snapshot isolation** for readers and **serialized writes**.

#### 3.1.1 Writers
- **Mutual Exclusion:** Only one writer can commit at a time. The `-shm` file contains a single-writer mutex.
- **Blocking:** A second writer must wait for the first to commit and complete its checkpoint (if any) before it can proceed.
- **Atomicity:** The commit is atomic with respect to the WAL. Either all changes are in the WAL, or none are.

#### 3.1.2 Readers
- **Non-Blocking Reads:** Multiple readers can read simultaneously, even while a writer is active.
- **Snapshot Isolation:** Each reader sees a consistent snapshot of the database as of the time it opened its read transaction.
    - If a reader opens a transaction *before* a writer commits, the reader will *not* see the writer's changes, even if the writer commits while the reader is still active.
    - If a reader opens a transaction *after* a writer commits, the reader *will* see the changes.
- **Read View:** The reader's view is defined by the **WAL index** at the time the read transaction began. The `-shm` file maintains a list of active readers and their "WAL index" (the last frame they have read).

#### 3.1.3 Reader and Checkpoint Interaction
- **Critical Constraint:** A reader cannot proceed past a point in the WAL where a checkpoint has overwritten pages in the main database, unless the reader has already read those pages from the WAL.
- **How it works:**
    1. When a checkpoint copies a page from the WAL to the main DB, it does not immediately invalidate that frame in the WAL.
    2. The `-shm` file tracks the "checkpointed frame index."
    3. A reader will read from the main database for pages that have been checkpointed, and from the WAL for pages that have not been checkpointed yet.
    4. **However**, if a passive checkpoint cannot proceed because a reader is holding a snapshot that includes a page in the WAL, the checkpoint stalls. The reader must finish before that page can be fully checkpointed (in a FULL checkpoint) or bypassed (in a PASSIVE checkpoint, which simply stops).
    5. **Data Loss Prevention:** A frame in the WAL is only considered "safe to discard" (i.e., the WAL file can be truncated) when *all* active readers have moved past that frame. This is managed by the `-shm` file's "read mark" mechanism.

### 3.2 Consistency Model
- **Serializable Isolation?** No. SQLite WAL mode provides **Snapshot Isolation** (SI) for readers. This is weaker than Serializability.
- **Anomalies:** Phantom reads can occur if a reader re-reads a range after a writer inserts new rows in that range (since the reader's snapshot is fixed at start, but re-reading might be a new transaction). However, within a single read transaction, the snapshot is immutable.
- **Write Skew:** Possible, as is standard for SI. SQLite does not provide full serializability without application-level locking.

### 3.3 Crash Recovery
- **Durability:** Once a commit is returned, the WAL frames are flushed to disk (via `fsync` or `fdatasync`, depending on platform and PRAGMAs).
- **Recovery Process:**
    1. On open, SQLite checks for a `-wal` file.
    2. It validates the WAL header checksum.
    3. It iterates through frames, validating each frame's checksum and salt.
    4. It applies all valid, committed frames to the main database.
    5. If a frame is invalid, it stops. This handles partial writes.
- **Limitation:** Recovery is linear in the size of the valid WAL. A very large WAL (e.g., hundreds of MBs) will result in slow startup times.

## 4. Key Limitations and Open Questions

### 4.1 Known Limitations
1.  **Single-Writer Only:** WAL mode does not support concurrent writers. This is a fundamental design choice to maintain strict consistency and simplify the recovery mechanism.
2.  **WAL File Growth:** If writers are slow or readers are long-lived, the WAL file can grow large. A passive checkpoint will not shrink it. Administrators must monitor `-wal` file size.
3.  **Remote File Systems:** WAL mode is **not recommended** for databases located on remote file systems (NFS, Samba, CIFS) or shared cloud storage. The `-shm` file relies on memory-mapping and file locking (`flock`/`fcntl`), which are unreliable or non-functional across network boundaries. This can lead to corruption.
    *   *Source: SQLite Documentation, "WAL File Locking".*
4.  **File System Requirements:** The file system must support `ftruncate()` and `fallocate()` (or similar) efficiently. Some file systems (e.g., certain versions of HFS+) had issues with the `-shm` file memory-mapping.
5.  **No Read-Only Mode for WAL:** A read-only connection to a WAL-mode database requires write access to the directory to create/update the `-shm` file. If the directory is read-only, WAL mode cannot be used.

### 4.2 Unverified or Ambiguous Areas
- **Precise Checkpoint Timing:** The exact moment a passive checkpoint stops can vary depending on the reader's progress and the OS's buffering behavior. The SQLite documentation states it "does not block," but in practice, a writer may be delayed if it must wait for a checkpoint to complete before it can write the next frame (though this is rare and typically only if the WAL is full).
- **Performance of `fdatasync` vs `fsync`:** SQLite uses `fdatasync` on Linux/Unix to avoid flushing metadata. The performance impact depends heavily on the underlying storage device. Some SSDs may not benefit, while HDDs gain significant latency reduction. *This is an empirical claim dependent on hardware; no single study covers all cases.*

## 5. Conclusion

SQLite's WAL mode is a sophisticated implementation of write-ahead logging that trades the simplicity of rollback journals for significantly higher concurrency. Its key innovations are:
1.  **Append-only log:** Avoids expensive file truncation/rewriting during commits.
2.  **Memory-mapped shared memory (`-shm`):** Enables efficient multi-process coordination without extensive system calls.
3.  **Snapshot isolation for readers:** Allows unlimited concurrent readers without blocking writers.
4.  **Passive checkpointing:** Prevents long-running readers from causing writer starvation.

**Critical Implementation Note:** Always ensure the database resides on a local, reliable file system. Do not use WAL mode over NFS. Monitor WAL file size and consider setting `PRAGMA wal_autocheckpoint` lower for high-write workloads to keep the file manageable.

## 6. Source Verification and Confidence

| Claim | Source | Confidence |
| :--- | :--- | :--- |
| WAL file header structure (32 bytes) | SQLite "WAL Format Specification" (2023) | High (Primary) |
| Frame header structure (24 bytes) | SQLite "WAL Format Specification" (2023) | High (Primary) |
| Checkpoint types (Passive/Full/Truncate) | SQLite `pragma.c` and documentation | High (Primary) |
| Default auto-checkpoint threshold (1000 pages) | SQLite documentation | High (Primary) |
| Single-writer, multi-reader concurrency | SQLite documentation | High (Primary) |
| NFS incompatibility | SQLite documentation | High (Primary) |
| Snapshot isolation behavior | SQLite documentation & source code review | High (Primary) |
| Performance impact of `fdatasync` | General database literature | Medium (Hardware-dependent) |

*No peer-reviewed academic papers were cited because SQLite's internals are best documented by the official project documentation and source code, which are considered primary and authoritative. Academic comparisons exist but are secondary and less precise on implementation details.*