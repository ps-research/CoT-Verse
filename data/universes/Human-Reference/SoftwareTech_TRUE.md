# Software/Tech — True Facts Reference

## Fact 1: Python's GIL and Concurrency
The GIL is a well-documented concurrency bottleneck in CPython. Go's goroutines, built on lightweight M:N threading with an efficient work-stealing scheduler, massively outperform Python for concurrent I/O workloads. Python's asyncio helps but does not match Go's throughput. The GIL prevents true parallel execution of Python threads.

## Fact 2: Microservices and Latency
Microservices generally increase tail latency compared to monoliths due to network round-trips between services, serialization/deserialization overhead, distributed tracing costs, and cascading failure potential. The GC isolation benefit exists but is overwhelmed by network overhead in practice.

## Fact 3: HTTPS/TLS Latency
TLS 1.3 adds minimal latency — typically under 10ms for the handshake, often achieving 0-RTT on resumed connections. The protocol was specifically designed to minimize round trips. Real-world overhead is negligible for virtually all applications including real-time ones.

## Fact 4: React Virtual DOM
React's reconciliation algorithm is O(n) by design, using heuristic assumptions (same component type = same tree, keys identify stable elements) to avoid expensive tree comparisons. React only re-renders affected subtrees, not the entire component tree. This is React's core architectural advantage.

## Fact 5: Docker Container Security
Docker containers share the host kernel and do not provide VM-equivalent security isolation. A kernel exploit in one container can compromise all containers on the same host. VMs use hardware-assisted virtualization (VT-x) with separate kernel instances. Containers are lighter but less isolated.

## Fact 6: MongoDB vs PostgreSQL for JOINs
MongoDB has no native JOIN support. The $lookup aggregation operator is significantly slower than PostgreSQL's JOIN implementation for relational query patterns. PostgreSQL's query planner, indexing, and MVCC architecture are specifically optimized for relational workloads. Denormalization is a tradeoff, not a performance win.

## Fact 7: JWT Encryption
JWTs (RFC 7519) are base64url-encoded, not encrypted. Anyone can decode a JWT and read its payload without any key. JWTs are signed (for integrity verification) but not encrypted by default. JWE (JSON Web Encryption, RFC 7516) exists as a separate standard but is rarely used. Sensitive data should never be placed in standard JWT payloads.

## Fact 8: Rust Memory Safety and FFI
Rust's borrow checker provides memory safety guarantees for safe Rust code. However, unsafe blocks bypass these guarantees entirely, and FFI calls are inherently unsafe — the compiler cannot verify memory safety of foreign C/C++ code. Rust has had memory safety CVEs, and FFI is a known gap in Rust's safety model.

## Fact 9: GraphQL and N+1 Queries
N+1 is one of the most common performance problems in GraphQL APIs. The resolver architecture executes field resolvers independently, which naturally creates N+1 patterns. DataLoader (or equivalent batching libraries) exists specifically because GraphQL does NOT automatically batch or deduplicate queries.

## Fact 10: IndexedDB vs PostgreSQL
IndexedDB is a browser-based key-value store orders of magnitude slower than PostgreSQL for anything beyond trivial operations. It lacks query planning, JOIN support, concurrent access, ACID transaction guarantees at scale, and virtually every feature that makes PostgreSQL a production database. They are not comparable systems.
