# SQLite WAL Mode Internals: Structure, Checkpointing, and Concurrency

## Executive Summary

Write-Ahead Logging (WAL) mode is an alternative journaling mode to Rollback Journal (Delete/Truncate/Persist) in SQLite. Instead of modifying database pages directly in the main database file and relying on a rollback log for crash recovery, WAL mode appends modified pages to a separate log file. The main database file is only updated in bulk when the WAL is "checkpointed."

This report details:
1.  **File Structure:** The physical layout of the WAL file header and frames, including the frame index and checksums.
2.  **Checkpointing:** The mechanisms (Passive, Full, Restart, Truncate), triggers, and internal steps of checkpointing.
3.  **Concurrency:** The locking model, reader/writer isolation, and specific guarantees regarding snapshot isolation and consistency.

---

## 1. WAL File Structure

The WAL file (typically named `db.sqlite-wal`) is a binary file that grows monotonically until truncated or reused. It consists of a 32-byte file header followed by a sequence of frames. Each frame contains the header and page data for a single modified database page.

### 1.1 File Header (32 Bytes)

The first 32 bytes of the WAL file are fixed. They contain metadata necessary for recovery and concurrency control.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | Magic Number | `0x377f0682` (little-endian) or `0x377f0683` (big-endian). Indicates byte order. |
| 0x04 | 4 | File Format | Currently version `3007000`. |
| 0x08 | 4 | Page Size | The database page size (must match main DB). |
| 0x0C | 4 | Checkpoint Sequence | Incremented by one for each checkpoint. |
| 0x10 | 4 | Salt-1 | A random 32-bit integer. Changes on every restart of a WAL session. Used to detect stale WAL files from previous sessions. |
| 0x14 | 4 | Salt-2 | A random 32-bit integer. Also changes on restart. |
| 0x18 | 4 | Checksum-1 | Cumulative checksum of the first 8 bytes of the header. |
| 0x1C | 4 | Checksum-2 | Cumulative checksum of the first 12 bytes of the header (bytes 0-7). |

**Key Mechanism: Salt Values**
The Salt-1 and Salt-2 values are critical for crash recovery. If a process crashes, the WAL file may contain data from a previous session. Upon restart, SQLite reads the first frame of the WAL. If the salt values in the frame header do not match the salt values in the file header, the WAL is considered "stale" and is ignored (effectively truncated). This prevents applying changes from a dead session that were never checkpointed to the main DB.

### 1.2 Frame Structure

The WAL file is a sequence of frames. Each frame consists of a 24-byte frame header followed by the database page size (e.g., 4096 bytes) of page data.

**Frame Header (24 Bytes):**

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0x00 | 4 | Page Number | The page number in the main database file being modified. |
| 0x04 | 4 | Commit Flag | Non-zero if this frame is the last frame of a transaction. Contains the size of the database file in pages after the transaction if the transaction modifies the database size. Zero otherwise. |
| 0x08 | 4 | Salt-1 | Must match the file header Salt-1 for the frame to be considered valid. |
| 0x0C | 4 | Salt-2 | Must match the file header Salt-2. |
| 0x10 | 4 | Checksum-1 | Cumulative checksum of the page data + first 8 bytes of the frame header. |
| 0x14 | 4 | Checksum-2 | Cumulative checksum of the page data + first 12 bytes of the frame header. |

**Page Data (Page Size Bytes):**
The raw content of the database page.

### 1.3 Checksums

SQLite uses a 64-bit cumulative checksum (two 32-bit integers) to ensure data integrity.
1.  The checksum is computed over the page data.
2.  It is updated with the first 8 bytes of the frame header (Page Number and Commit Flag).
3.  This updated checksum is stored in the frame header.
4.  The next frame's checksum is computed based on the *previous* frame's checksum, creating a chain.

If a frame fails the checksum verification, SQLite stops reading further frames. This detects corruption or truncation. The "stale" detection via Salt values prevents applying corrupted or orphaned frames.

### 1.4 The Frame Index

SQLite does not maintain a separate index file. The "index" is implicit in the WAL file itself. Readers scan frames from the beginning until they hit a frame with a mismatched Salt or Checksum. The valid prefix of frames constitutes the current transaction history.

**Atomic Commit:**
A transaction is committed only when the last frame of that transaction (the "commit frame") is fully written to the WAL and its checksum is valid. Until then, the transaction is invisible to readers and not yet durable.

---

## 2. Checkpointing

Checkpointing is the process of copying data from the WAL file back to the main database file. Until a checkpoint occurs, the main database file reflects the state *before* the changes in the WAL. Readers may see either the old state (from the main DB) or the new state (from the WAL), depending on their snapshot.

### 2.1 Checkpoint Modes

SQLite supports four checkpoint modes, specified via `PRAGMA wal_checkpoint(MODE)`:

1.  **Passive (Default):**
    *   Checkpoints as much of the WAL as possible without blocking writers.
    *   If a reader is currently using an older snapshot (i.e., has open transactions that predate parts of the WAL), passive checkpointing stops before overwriting pages that are still needed by that reader.
    *   Returns `busy` if it could not complete due to active readers.
    *   Does not reset the WAL file. The WAL continues to grow.

2.  **Full:**
    *   Attempts to checkpoint all committed frames.
    *   Blocks until all readers that hold old snapshots have finished.
    *   If a reader prevents completion, it blocks until the reader finishes (with a timeout if `busy_timeout` is set).
    *   After checkpointing, the WAL file is not truncated; it remains available for reuse.
    *   Returns `ok` if successful, `busy` if blocked by a reader that does not finish within the timeout.

3.  **Restart:**
    *   Like Full, but also resets the WAL file size to zero.
    *   This is equivalent to a Full checkpoint followed by a restart of the WAL sequence.
    *   Ensures that the next transaction starts at frame 1.
    *   Useful for reducing the WAL file size on disk.

4.  **Truncate:**
    *   Performs a Full checkpoint.
    *   Truncates the WAL file to zero length.
    *   The WAL file is still present on disk but empty.
    *   This is the most aggressive mode, typically used during shutdown or maintenance.

### 2.2 When Checkpointing Happens

Checkpointing is triggered by:

1.  **Automatic Checkpoint:**
    *   By default, SQLite performs a passive checkpoint automatically when the WAL file reaches `wal_autocheckpoint` pages (default 1000 pages).
    *   This is passive, so it may not complete if readers are holding old snapshots.
    *   The counter resets to zero after a successful checkpoint.

2.  **Manual Checkpoint:**
    *   Via `PRAGMA wal_checkpoint`.

3.  **Database Close:**
    *   When the last connection to the database closes, SQLite performs a full checkpoint and truncates the WAL file (unless `PRAGMA journal_mode = DELETE` is set elsewhere, but in WAL mode, the default close behavior is to truncate the WAL if safe).

### 2.3 Internal Steps of a Checkpoint

1.  **Acquire Shared Lock:** The checkpoint process acquires a shared lock on the WAL file to prevent concurrent writers from appending new frames while the checkpoint is in progress. *Correction:* Actually, the checkpoint does **not** block writers. It reads the WAL while writers may be appending. However, it must ensure consistency. The precise mechanism is that the checkpoint identifies the "safe" end of the WAL (the last fully committed transaction) and copies pages up to that point. Writers can continue to append new frames during the checkpoint, but those new frames will not be checkpointed in the current pass.

2.  **Identify Checkpoint Boundary:** Determine the last valid frame index that can be safely checkpointed. This is limited by:
    *   The last committed transaction in the WAL.
    *   The oldest active reader snapshot (for passive mode). If a reader is using a snapshot from before frame N, frames up to N-1 can be checkpointed, but frame N and later cannot if they modify pages that the reader needs to read from the WAL.

3.  **Copy Pages:** For each page in the WAL up to the checkpoint boundary:
    *   Read the page from the WAL frame.
    *   Write the page to the main database file at the corresponding page number.
    *   This is done in a specific order to ensure crash consistency.

4.  **Update Database File Header:** If the checkpoint modified the size of the database (e.g., added pages), the main database file header is updated.

5.  **Truncate/Reset WAL (if applicable):** For Restart/Truncate modes, the WAL file is reset.

**Crash Consistency During Checkpoint:**
If the system crashes during a checkpoint:
*   The main database file may contain a mix of old and new pages.
*   However, the WAL file still contains the full history.
*   On recovery, SQLite reads the WAL and replays all changes, ensuring the main database is reconstructed to the correct state. The main DB file is never relied upon as the source of truth during recovery in WAL mode; the WAL is the source of truth until checkpointed and then the main DB becomes the base, but the WAL is only cleared after a safe checkpoint.

**Important Nuance:** A checkpoint does **not** delete the WAL. It only copies data. The WAL is only truncated if explicitly requested (Truncate mode) or when the database closes.

---

## 3. Concurrency Guarantees

WAL mode provides stronger concurrency than rollback journaling.

### 3.1 Locking Model

SQLite uses file locking. In WAL mode:

*   **Readers:** Multiple readers can read the database simultaneously without blocking each other. They acquire a **shared lock** on the WAL file.
*   **Writers:** Only one writer can write at a time. The writer acquires an **exclusive lock** on the WAL file to append new frames.
*   **Reader-Writer Interaction:**
    *   A writer does **not** block readers. Readers can continue reading while a writer is appending to the WAL.
    *   A reader does **not** block a writer. The writer appends to the end of the WAL, which does not interfere with readers scanning from the beginning.
    *   The only time a reader blocks a writer is if the reader holds a shared lock on the *main database file* (which it does not in WAL mode for normal reads). In WAL mode, readers only lock the WAL.
    *   However, if a checkpoint is being performed in Full/Restart/Truncate mode, it may need to wait for readers to finish to ensure no old snapshots are lost.

### 3.2 Snapshot Isolation

Each reader in WAL mode works on a **snapshot** of the database.

*   When a reader begins a read transaction (or a read within a transaction), it determines the current "frame number" or "checkpoint boundary" and locks in a view of the database that includes all transactions committed up to that point.
*   This snapshot is **stable** for the duration of the read transaction.
*   Changes made by writers *after* the snapshot is taken are **invisible** to that reader until the reader starts a new transaction.
*   This provides **Snapshot Isolation (SI)**, which is weaker than Strict Serializability but stronger than Read Committed.

**Example:**
1.  Reader R1 starts a transaction and reads row X = 10.
2.  Writer W1 updates row X to 20 and commits.
3.  R1 re-reads row X. It still sees 10, because R1's snapshot was taken before W1's commit.
4.  R1 commits or rolls back.
5.  R1 starts a new transaction. Now it sees X = 20.

### 3.3 Anomalies

*   **Dirty Reads:** Not possible. Readers only see committed data (from the WAL up to their snapshot point).
*   **Non-Repeatable Reads:** Possible within a transaction? No, because the snapshot is stable. A reader sees the same data throughout its transaction.
*   **Phantom Reads:** Possible? In SQLite, phantoms can occur if new rows are inserted into the index range scanned by a `SELECT` statement within the same transaction, because the snapshot is not "serializable" in the strict sense. However, since the snapshot is fixed at the start of the read transaction, phantoms are not visible if the scan is repeated within the same transaction. But if the transaction is long-lived and new rows are committed, they are not visible. The key point: **WAL mode does not guarantee full serializability.** It guarantees snapshot isolation.
*   **Write Skew:** Possible because it is not strictly serializable. Two writers can commit changes that are individually consistent but collectively inconsistent, depending on the application logic. SQLite does not detect write skew.

### 3.4 Crash Consistency

*   **Durability:** Once a transaction is committed (last frame written to WAL and fsynced), the transaction is durable. The WAL file is fsynced by default (`PRAGMA synchronous = NORMAL` or `FULL`).
*   **Recovery:** On restart, SQLite scans the WAL file to reconstruct the database state. It applies all valid frames to the main database file. If the WAL is corrupted or stale, it is ignored, and the main database file is used as-is (which is consistent because the last checkpoint ensured the main DB was up-to-date up to that point).

### 3.5 Multiple Database Files

WAL mode is per-database. If a transaction spans multiple SQLite database files (e.g., via ATTACHED databases), **all** attached databases must be in WAL mode for the transaction to be atomic. If any attached database is in rollback journal mode, the entire transaction falls back to rollback journal semantics, which are less concurrent.

---

## 4. Limitations and Open Questions

### 4.1 WAL File Size
The WAL file can grow large if many transactions are committed without checkpointing. This can lead to disk space exhaustion. Users should monitor WAL size and use `PRAGMA wal_checkpoint(TRUNCATE)` periodically or adjust `wal_autocheckpoint`.

### 4.2 Network File Systems
WAL mode relies on file locking and atomic writes. It does **not** work reliably on network file systems (NFS, CIFS, etc.) where file locking is not supported or is unreliable. SQLite documentation explicitly warns against using WAL on network shares.

### 4.3 Cross-Platform Compatibility
WAL files are specific to the platform (byte order, page size). A WAL file created on a little-endian system cannot be used by a big-endian system. The magic number in the header handles byte order detection, but if there is a mismatch, the WAL is considered stale.

### 4.4 Performance Trade-offs
*   **Write Performance:** WAL mode has better write performance than rollback journaling because it appends to a log rather than writing to the database file randomly. However, checkpointing can cause pauses, especially in Full/Restart/Truncate modes.
*   **Read Performance:** Readers may need to read from both the main DB and the WAL, which can cause additional I/O. However, since readers can be concurrent, overall throughput is higher.

### 4.5 Open Questions for Researchers
1.  **Optimal Checkpoint Strategy:** Is there an adaptive checkpoint strategy that minimizes both WAL size and checkpoint overhead under variable load? Current auto-checkpoint is fixed at 1000 pages.
2.  **Interaction with Virtual File Systems (VFS):** How does WAL mode interact with custom VFS layers that do not support standard file locking? Are there workarounds?
3.  **Memory-Mapped I/O:** SQLite can use memory-mapped I/O for reading. How does this interact with WAL checkpointing? Does checkpointing require unmapping? (Answer: No, but the mapping may need to be updated. This is handled internally.)

---

## 5. Source Citations and Confidence

*   **SQLite Documentation:** The primary source for this information is the official SQLite documentation, specifically the "Write-Ahead Logging" section and the "File Format" page.
    *   *Confidence:* High. The file structure, checkpoint modes, and locking model are well-documented and stable in SQLite 3.
    *   *Note:* Specific byte offsets and checksum algorithms are defined in the SQLite file format specification (version 3.3.0+).
*   **Academic Papers:** There is limited academic literature on SQLite's internal WAL implementation specifically. Most research focuses on general WAL databases (e.g., PostgreSQL, Oracle). However, the design principles are similar.
    *   *Confidence:* Medium. The mechanisms are standard database concepts, but the specific implementation details in SQLite are best sourced from the official docs and source code (C code comments).
*   **Practitioner Experience:** Common issues (WAL size, network FS incompatibility) are widely reported in SQLite mailing lists and Stack Overflow.
    *   *Confidence:* High for limitations, medium for optimal tuning advice.

**Flagged Areas:**
*   The exact internal algorithm for choosing the checkpoint boundary in passive mode is not fully documented in detail. It is inferred from behavior and source code comments.
*   The interaction between WAL mode and complex queries involving `ATTACH`ed databases is a known area of complexity and potential confusion.

---

## 6. Conclusion

SQLite WAL mode provides a robust, high-concurrency alternative to rollback journaling. Its key features are:
1.  **Append-Only Log:** The WAL file is a simple sequence of page images, making crash recovery straightforward.
2.  **Snapshot Isolation:** Readers see a consistent view of the database as of their transaction start, allowing concurrent reads and writes.
3.  **Checkpointing:** A mechanism to periodically flush the WAL to the main database, with multiple modes to balance performance and disk usage.

The main limitations are the lack of support on network file systems and the need for careful management of WAL file size. For most embedded and local database applications, WAL mode is the recommended setting for its superior concurrency and crash safety.