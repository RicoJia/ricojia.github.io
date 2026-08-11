---
layout: post
title: C++ - [OOP] Destruction
date: 2023-03-02 13:19
subtitle: Destruction Ordering, Cyclic Shared Pointer Skips Dtor
comments: true
header-img: img/post-bg-alitrip.jpg
tags:
  - C++
---
## Destruction Ordering

The destruction order is `Derived Class -> Derived class members -> base class`, which is the inverse order of construction: `base class -> Derived class members -> Derived Class`

```cpp
#include <iostream>

struct A
{
    ~A() { std::cout << "A\n"; }
};

struct C
{
    ~C() { std::cout << "C\n"; }
};

struct B : public A
{
    C c1;                   // data-member declared *after* any implicit A sub-object

    ~B() { std::cout << "B\n"; }
};

int main()
{
    {
        B obj;              // construct a B on the stack
    }                       // scope ends → destructors run

    return 0;
}
```

---
## Cyclic Shared Pointer Skips Dtor 

Destruction ordering only tells you what happens *when* a destructor runs. It says nothing about **whether** it runs at all. With `shared_ptr`, that is a separate question:

```cpp
int main(){
    auto node = std::make_shared<Node>();
}
```

`node` is a local variable with **automatic storage duration** — informally, a "stack object." But `node` is only the `shared_ptr` *handle*. The `Node` itself lives on the heap. Two objects, two lifetimes:

```text
Automatic storage               Dynamic storage
─────────────────               ───────────────
shared_ptr node  ──────────────► Node object
(local handle)                   (heap allocation)
 dtor ALWAYS runs                dtor runs only at refcount 0
```

At the end of scope, the handle's destructor always runs and decrements the count. The heap object is destroyed **only if that count reaches zero**. So a `shared_ptr` guarantees you a *decrement*, not a *destruction*.

### When the count never reaches zero

The usual way to get stuck is a cycle — two objects that own each other:

```cpp
#include <iostream>
#include <memory>

struct Worker;

struct Node : std::enable_shared_from_this<Node> {
    std::shared_ptr<Worker> worker;          // Node owns Worker
    ~Node() { std::cout << "~Node\n"; }
    void start();
};

struct Worker {
    std::shared_ptr<Node> node;              // ...and Worker owns Node back
    explicit Worker(std::shared_ptr<Node> n) : node(std::move(n)) {}
    ~Worker() { std::cout << "~Worker: flushing report\n"; }
};

void Node::start() { worker = std::make_shared<Worker>(shared_from_this()); }

int main() {
    auto node = std::make_shared<Node>();
    node->start();
}
```

This program prints **nothing**. Walk the counts:

| Point in `main` | count(Node) | count(Worker) |
| --- | --- | --- |
| after `make_shared<Node>()` | 1 — local handle | 0 |
| after `start()` | **2** — local + `worker->node` | 1 — `node->worker` |
| end of scope, local handle destroyed | **1** — still held by `worker->node` | 1 |

The local handle's destructor runs and drops the count to 1, not 0. So `~Node` never runs, `node->worker` is never released, `~Worker` never runs either. Each object is kept alive purely by the other.

### "But the process is exiting — doesn't everything get destroyed?"

Memory, yes. Destructors, no. These are different things, and only one of them happens:

| At process exit                      | Destructors                                    |
| ------------------------------------ | ---------------------------------------------- |
| Automatic (stack) objects in `main`  | **run**                                        |
| Static / thread-local objects        | **run**, via `atexit`                          |
| Heap objects with a nonzero refcount | **never run** — the OS just reclaims the pages |

If a destructor only frees memory, a cycle is harmless in practice: the kernel reclaims everything when the process dies. That's exactly why this bug hides so well — nothing crashes, nothing warns.

It stops being harmless the moment a destructor has an **effect outside the process**:

```cpp
~Worker() {
    thread_.join();        // never happens → thread killed mid-flight
    ofs_ << summary();     // never happens → file silently never written
}
```

Reclaiming memory does not join a thread and does not flush a stream that was never written to. You get a clean exit and a missing report.

### Fixes

**Break the cycle.** Ownership should point one way. If the `Node` owns the `Worker`, the `Worker` cannot outlive it, so it does not need to own anything back:

```cpp
struct Worker {
    Node* node;                              // non-owning back-reference
    explicit Worker(Node* n) : node(n) {}
    ~Worker() { std::cout << "~Worker: flushing report\n"; }
};

void Node::start() { worker = std::make_shared<Worker>(this); }   // plain `this`
```

`Node` no longer needs `enable_shared_from_this` at all. The same `main` now prints:

```text
~Node
~Worker: flushing report
```

Use `std::weak_ptr<Node>` instead of a raw pointer when the back-reference might genuinely outlive the owner, and `lock()` it before each use.

**If you cannot break it,** stop relying on the destructor and call the cleanup explicitly, while the process is still alive:

```cpp
int main() {
    auto node = std::make_shared<Node>();
    node->start();
    node->finalize();     // join threads, write reports — do it here
}
```

That works, but it is a workaround: `finalize()` is now a second destructor that the compiler will never call for you, and every early return or exception path has to remember it. Prefer breaking the cycle.

### Leak detectors help less than you would hope

A cycle is not reported as "still reachable" — `LeakSanitizer` classifies it as an **indirect leak**, since each object is only kept alive by another leaked object:

```text
Indirect leak of 48 byte(s) in 1 object(s)     <- the Node
Indirect leak of 32 byte(s) in 1 object(s)     <- the Worker
SUMMARY: AddressSanitizer: 80 byte(s) leaked in 2 allocation(s).
```

But compile the exact `main` from earlier with `-fsanitize=address` and LSan reports **nothing**. The destroyed handle's bytes are still lying in `main`'s dead stack frame, and LSan scans the stack *conservatively* — it sees something that looks like a pointer to the `Node` and concludes the object is still reachable.

To make the leak visible, move the work into a function and overwrite the frame:

```cpp
void run()   { auto node = std::make_shared<Node>(); node->start(); }
void scrub() { volatile char buf[4096]; for (int i = 0; i < 4096; ++i) buf[i] = 0; }

int main() { run(); scrub(); }      // now LSan reports the 80 bytes
```

Which is the real lesson: a clean sanitizer run does not prove you have no cycle. Reason about ownership direction instead of relying on the tool to find it.