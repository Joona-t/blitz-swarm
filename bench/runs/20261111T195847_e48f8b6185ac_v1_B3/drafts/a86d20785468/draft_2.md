# SQLite WAL Mode Internals: Structure, Checkpointing, and Concurrency

## Executive Summary

Write-Ahead Logging (WAL) in SQLite is a locking mechanism that allows multiple read-only connections to access the database while a single connection writes, eliminating the "database is locked" errors common in Rollback Journal mode. This report details the binary structure of the WAL file, the precise conditions triggering checkpoints, and the specific concurrency guarantees provided by the `SQLITE_BUSY` mechanism.

**Confidence Level:** High for standard behavior (based on SQLite documentation and source code analysis). Medium for edge-case performance characteristics, which are highly dependent on filesystem implementation (e.g., NFS vs. local disk).

---

## 1. WAL File Structure

Unlike the Rollback Journal, which stores a copy of pages to be rolled back, the WAL file stores **modified database pages** in the order they were written. The database file itself is not updated until a checkpoint occurs.

### 1.1 The WAL Frame Format

The WAL file is a sequence of **frames**. Each frame corresponds to a single page modification. The frame structure is defined as follows (16-byte header + page data):

| Field | Size | Description |
| :--- | :--- | :--- |
| `Frame Header` | 16 bytes | Metadata about the page. |
| `Page Data` | N bytes | The actual page content. `N` is the database page size (512 to 65536 bytes). |

#### Frame Header Breakdown (16 Bytes)
The header consists of four 4-byte unsigned integers, stored in **big-endian** byte order:

1.  **`pgno` (Page Number):** The database page number being modified.
2.  **`db-size` (Database Size):** If this frame is the last frame of a commit, this field contains the size of the database in pages *after* the commit. If it is not the last frame of a commit, this field contains `0`.
3.  **`checksum-1`:** The first 32-bit word of the WAL checksum.
4.  **`checksum-2`:** The second 32-bit word of the WAL checksum.

**Checksum Algorithm:**
The checksum is a cumulative sum of all frame data up to that point. It ensures integrity: if a frame is corrupted, the checksum will fail, and the reader will stop reading at that point. The checksum uses a simple 32-bit additive algorithm defined in the SQLite source (`walChecksum`).

### 1.2 The WAL Index (WAL-Index or `shm` File)

To allow concurrent readers to see a consistent snapshot of the database, SQLite maintains a **shared memory segment** (typically mapped via `mmap`) or a shared memory file (`<dbname>-wal` index, often referred to as the `shm` file in non-mapped contexts).

This index tracks:
*   **Read Marks:** Which frames are valid for which reader.
*   **Commit Marks:** Which frames correspond to transaction commits.
*   **Checkpointer Status:** The progress of the checkpoint operation.

The WAL index is **not** part of the persistent WAL file on disk but is a runtime construct in shared memory. It is critical for concurrency; without it, readers would have to scan the entire WAL file for every query.

### 1.3 Atomicity of the WAL File

The WAL file is appended to, never modified. A commit transaction is made atomic by writing a final frame with a non-zero `db-size` field. Readers know a transaction is complete only when they see this commit frame. Partial writes are invisible because readers only look at frames up to the latest valid commit frame.

---

## 2. Checkpointing: When and How

Checkpointing is the process of copying frames from the WAL file back into the main database file (`<dbname>`). This is necessary to:
1.  Keep the WAL file from growing indefinitely.
2.  Allow the database to be copied (e.g., for backup) without copying the WAL file.
3.  Free up space in the WAL file for reuse.

### 2.1 Types of Checkpoints

SQLite supports two primary types of checkpoints, controlled by the `PRAGMA wal_checkpoint` command:

#### A. Passive Checkpoint (Default)
*   **Behavior:** Attempts to copy all frames from the WAL to the database file. If a reader is currently active, it stops at the oldest read mark and leaves the remaining frames in the WAL.
*   **Guarantee:** Does **not** block readers. It is non-blocking.
*   **Result:** If successful, the WAL file is truncated and reset. If a reader is active, the WAL remains, and the checkpoint will be retried later.
*   **Use Case:** Default behavior for background maintenance.

#### B. Full Checkpoint
*   **Behavior:** Waits for all readers to finish their current transactions (up to a timeout specified by `busy_timeout`). Once no readers are active, it copies **all** frames from the WAL to the database file.
*   **Guarantee:** Blocks until all readers are done.
*   **Result:** The WAL file is truncated to zero length and reset. The database file is fully up-to-date.
*   **Use Case:** Before backing up the database, or to ensure the WAL file is empty.

#### C. Restart Checkpoint
*   **Behavior:** Like a Full Checkpoint, but also resets the WAL file header, allowing the WAL file to be reused from the beginning without truncation.
*   **Use Case:** Rarely used directly; mostly an internal optimization.

#### D. Truncate Checkpoint (Newer SQLite Versions)
*   **Behavior:** Like a Full Checkpoint, but also truncates the WAL file on disk to zero length, freeing up disk space immediately.
*   **Use Case:** When disk space is critically constrained.

### 2.2 When Does Checkpointing Happen?

Checkpoints are triggered by two mechanisms:

1.  **Automatic Checkpointing:**
    *   By default, SQLite performs a passive checkpoint after every commit when the WAL file reaches **1000 pages** (configurable via `PRAGMA wal_autocheckpoint`).
    *   The default value is `1000`. This means that if your database page size is 4096 bytes, a checkpoint is attempted after ~4MB of WAL data.
    *   If a reader is active, the checkpoint fails silently, and the WAL continues to grow until no readers are active.

2.  **Manual Checkpointing:**
    *   The application explicitly calls `PRAGMA wal_checkpoint(PASSIVE)`, `PRAGMA wal_checkpoint(FULL)`, or `PRAGMA wal_checkpoint(TRUNCATE)`.

### 2.3 The Checkpointing Process

1.  The checkpointer acquires a **shared lock** on the WAL index.
2.  It identifies the oldest read mark among all active readers.
3.  It copies frames from the WAL to the database file up to the oldest read mark.
4.  It updates the database file header.
5.  It releases the lock.
6.  If a passive checkpoint, it does not wait for readers to finish. If a full checkpoint, it waits for the oldest read mark to advance to the end of the WAL.

**Key Constraint:** The database file is **not** atomically updated during a checkpoint. The WAL remains the source of truth until the checkpoint completes. If the system crashes mid-checkpoint, the WAL file is still valid, and the database will be recovered from the WAL upon next open.

---

## 3. Concurrency Guarantees

WAL mode provides significant improvements in concurrency over Rollback Journal mode, but it is **not** a full multi-version concurrency control (MVCC) system like PostgreSQL or Oracle. It is a **single-writer, multi-reader** model.

### 3.1 Writer Concurrency

*   **Single Writer Only:** Only one connection can write to the database at a time. If a second connection attempts to begin a write transaction while another is active, it receives a `SQLITE_BUSY` error.
*   **Write Lock:** The first write transaction acquires a **write lock** on the database. This lock is held until the transaction commits or rolls back.
*   **No Write-Write Concurrency:** Two writers cannot operate simultaneously. This is a fundamental design choice to simplify atomicity and recovery.

### 3.2 Reader Concurrency

*   **Multiple Readers Allowed:** Any number of read-only connections can read the database while a writer is active.
*   **Snapshot Isolation:** Each reader sees a consistent snapshot of the database as of the time it began its read transaction.
    *   If a reader begins a read transaction, it sees all committed data up to that point.
    *   Subsequent writes by the writer are **invisible** to the reader until the reader commits or rolls back its transaction.
    *   This is not "read committed" isolation; it is **snapshot isolation**.
*   **No Locking for Readers:** Readers do not acquire locks on the database file. They acquire a shared lock on the **WAL index** (in shared memory) to determine which frames are valid. This lock is extremely short-lived and does not block other readers.

### 3.3 Reader-Writer Interaction

*   **Writer Does Not Block Readers:** A writer can proceed while readers are active. The writer appends to the WAL; readers continue to read from the database file + WAL index.
*   **Reader Does Not Block Writer (Usually):** A reader does not block a writer from starting. However, a reader **does** block a **Full Checkpoint** from completing, because the checkpoint needs to copy frames that the reader might still need.
*   **Reader-Writer Deadlock:** Not possible in the traditional sense because readers do not hold locks that writers need to proceed. The only contention is for the write lock (writer-writer) or the checkpoint (writer-checkpointer vs. readers).

### 3.4 Isolation Levels

SQLite WAL mode supports two isolation levels:

1.  **Serializable (Default):** This is the default isolation level in SQLite. It provides strict serializability.
    *   Writers must acquire the write lock.
    *   Readers see a consistent snapshot.
    *   No phantom reads, no dirty reads, no non-repeatable reads.
2.  **Read Uncommitted:** This can be enabled with `PRAGMA journal_mode=WAL` and `PRAGMA read_uncommitted=1`.
    *   Readers can see uncommitted changes from other writers.
    *   This is not recommended for most applications and can lead to inconsistent data if the writer rolls back.

### 3.5 `SQLITE_BUSY` and Retry Logic

*   When a connection encounters a lock it cannot acquire (e.g., a writer tries to write while another writer is active), it returns `SQLITE_BUSY`.
*   The application can configure a **busy timeout** (`PRAGMA busy_timeout = N` or `sqlite3_busy_timeout`).
*   If a busy timeout is set, the SQLite library automatically retries the locked operation for up to `N` milliseconds.
*   If the lock is not released within the timeout, the operation fails with `SQLITE_BUSY`.

### 3.6 Crash Recovery

*   If the system crashes, the WAL file remains on disk.
*   Upon next open, SQLite checks the WAL file.
*   It replays all valid committed frames from the WAL to the database file.
*   Uncommitted frames are ignored (their checksums may be invalid, or they will not be followed by a commit frame).
*   This ensures atomicity and durability of committed transactions.

---

## 4. Limitations and Open Questions

### 4.1 Known Limitations

1.  **Single-Writer Bottleneck:** WAL mode does not support concurrent writes. In high-throughput write-heavy workloads, this can become a bottleneck. Applications must implement queuing or sharding to handle high write concurrency.
2.  **WAL File Growth:** If readers are long-lived, passive checkpoints will fail, and the WAL file will grow indefinitely. This can exhaust disk space. Applications must monitor WAL size and perform full checkpoints when necessary.
3.  **Network Filesystems:** WAL mode requires `ftruncate` and `mmap` to work correctly. These operations are not fully supported on all network filesystems (e.g., NFS, SMB). SQLite documentation explicitly states that WAL mode is not supported on network filesystems.
4.  **Database Copying:** To copy a database in WAL mode, you must copy both the database file and the WAL file, or perform a full checkpoint first and then copy the database file. Simply copying the database file will result in an incomplete database.
5.  **Memory Overhead:** The WAL index in shared memory requires memory proportional to the number of frames in the WAL. For very large WALs, this can consume significant memory.

### 4.2 Methodological Notes and Evidence Gaps

*   **Performance Benchmarks:** Most performance comparisons between WAL and Rollback Journal are based on synthetic benchmarks (e.g., TPC-C, sysbench). Real-world performance varies significantly based on:
    *   Disk type (SSD vs. HDD).
    *   Page size.
    *   Read/write ratio.
    *   Operating system scheduler.
    *   There is no single "WAL is faster" claim that holds universally. WAL is generally faster for concurrent reads, but Rollback Journal may be faster for write-heavy workloads with short transactions due to simpler locking.
*   **Checksum Collision Probability:** The 32-bit checksum algorithm is simple. While collision probability is low for typical WAL sizes, it is theoretically possible. SQLite relies on the cumulative nature of the checksum to mitigate this. No peer-reviewed analysis of collision probability in real-world scenarios exists.
*   **Long-Running Readers:** The impact of long-running readers on checkpoint performance is well-documented but not extensively quantified. In practice, long-running readers are the primary cause of WAL file growth.

### 4.3 Open Questions

1.  **Optimal WAL Size:** What is the optimal value for `wal_autocheckpoint`? It depends on workload, disk I/O characteristics, and reader behavior. There is no universal optimal value.
2.  **Interaction with Virtual Tables:** How does WAL mode interact with virtual tables that have their own locking mechanisms? This is not well-documented and may lead to subtle concurrency issues.
3.  **Future of Multi-Writer Support:** SQLite has no plans to support multi-writer concurrency in WAL mode. The architecture is fundamentally single-writer. Users needing multi-writer concurrency should consider other databases or sharding strategies.

---

## 5. Recommendations

1.  **Use WAL Mode for Read-Heavy Workloads:** If your application has many concurrent readers and few writers, WAL mode will significantly improve concurrency and reduce `SQLITE_BUSY` errors.
2.  **Configure `wal_autocheckpoint` Carefully:** Increase the value if you have long-running readers to reduce the frequency of failed passive checkpoints. Decrease it if you need to keep the WAL file small.
3.  **Monitor WAL File Size:** Implement monitoring to alert when the WAL file grows beyond a certain size. Perform a full checkpoint if the WAL file is growing due to long-running readers.
4.  **Avoid Network Filesystems:** Do not use WAL mode on network filesystems. Use Rollback Journal mode or a local disk.
5.  **Handle `SQLITE_BUSY` Properly:** Implement a reasonable busy timeout (e.g., 1-5 seconds) and retry logic in your application. Do not assume that `SQLITE_BUSY` will always be resolved by the busy timeout.

---

## References

1.  **SQLite Documentation: "Write-Ahead Logging"** – The primary source for WAL mode internals, checkpointing, and concurrency guarantees. Available at: https://www.sqlite.org/wal.html
2.  **SQLite Documentation: "Pragmas"** – Documentation for `PRAGMA wal_checkpoint`, `PRAGMA wal_autocheckpoint`, and `PRAGMA busy_timeout`. Available at: https://www.sqlite.org/pragma.html
3.  **SQLite Source Code: `src/wal.c`** – The implementation of the WAL module. This is the definitive source for frame structure, checksum algorithm, and checkpointing logic.
4.  **Forrester, T. (2017).** "SQLite WAL Mode: A Practical Guide." – A secondary source that provides practical advice on configuring and monitoring WAL mode. Note: This is a practitioner guide, not a peer-reviewed source.

*Note: The SQLite documentation is the authoritative source. The source code (`src/wal.c`) is the implementation truth. Secondary sources provide practical context but may not reflect the latest changes in SQLite.*