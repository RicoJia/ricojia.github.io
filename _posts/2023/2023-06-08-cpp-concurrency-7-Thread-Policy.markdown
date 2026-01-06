---
layout: post
title: C++ - [Concurrency 7]
date: 2023-06-08 13:19
subtitle:
comments: true
header-img: img/post-bg-unix-linux.jpg
tags:
  - C++
---
## Thread Policy

Mental Model: each CPU repeatedly picks "the next runnable thread" to run. Different scheduling policies define different `pick_next()` rules.

This is the core idea behind Linux’s normal scheduler (**CFS = Completely Fair Scheduler**, used for `SCHED_OTHER`).

Instead of “everyone gets the same fixed time slice”, CFS tries to be fair by tracking a per-thread number called **virtual runtime** (`vruntime`):

- Every time a thread runs, its `vruntime` increases.

- The scheduler tends to pick the runnable thread with the **smallest vruntime** (the one that has had the least “fair share” so far).

So `vruntime` is like:

> “How much CPU time have you had, adjusted for your priority?”

Adjusted means: higher-priority (less “nice”) threads accumulate vruntime more slowly → they get more CPU over time.

As a clear and concise way to explain, here is some pseudocode for each thread's policy and a CPU.  Some abbreviations I'm using below:

- RT: real time.  In linux they are `SCHED_FIFO` and `SCHED_RR` , which has explicit priorities (1-99 on Linux)
- RR `SCHED_RR`: round Robin. Among threads with the **same real-time priority**, each gets a **time slice**.
- `SCHED_FIFO`: FIFO has **no time slice**. A thread can run forever until it blocks/yields (unless a higher-priority RT thread becomes runnable).
- vruntime: virtual run time - the count of total runtime of a thread
- **Nice**:  the user-facing priority control for normal threads (`SCHED_OTHER`, also affects `SCHED_BATCH`). It doesn’t force absolute priority like RT does. It changes **CPU share** under CFS. Thread A nice 0, Thread B nice 10 → A tends to get **more CPU** than B when both are runnable.

```cpp
# Linux-ish scheduling policies (mental model)

enum Policy {
  OTHER,     # normal time-sharing (CFS)
  FIFO,      # real-time: highest priority wins, no time slice
  RR,        # real-time: highest priority wins, round-robin within same priority
  BATCH,     # like OTHER but favors throughput over latency
  IDLE,      # runs only when nothing else is runnable
  DEADLINE   # earliest deadline first (specialized)
}

enum State { RUNNABLE, BLOCKED }

struct Thread {
  Policy policy;
  State  state;

  # Real-time knobs (FIFO/RR)
  int rt_priority;        # higher wins (Linux: typically 1..99)
  time rr_slice_left;     # only meaningful for RR

  # Normal scheduling knobs (OTHER/BATCH)
  int nice;               # lower nice => more CPU share
  time runtime;           # raw CPU time consumed
  time vruntime;          # "virtual runtime" used by CFS (scaled by nice)

  # Deadline knobs (DEADLINE)
  deadline_params dl;     # e.g. runtime, period, deadline
}

# ----------------------------
# Core scheduler: pick next
# ----------------------------

function pick_next(runnable_threads):
  # 1) DEADLINE: earliest deadline first (if any eligible)
  if exists t in runnable_threads where t.policy == DEADLINE and eligible(t):
    return earliest_deadline_first(runnable_threads where policy == DEADLINE and eligible)

  # 2) Real-time (FIFO/RR): highest priority wins
  if exists t in runnable_threads where t.policy in {FIFO, RR}:
    p = max rt_priority among runnable_threads where policy in {FIFO, RR}
    candidates = runnable_threads where policy in {FIFO, RR} and rt_priority == p

    # FIFO runs until it blocks/yields (or preempted by higher prio RT)
    if exists t in candidates where t.policy == FIFO:
      return fifo_pick(candidates where policy == FIFO)   # e.g. oldest FIFO at that priority

    # Otherwise RR at that priority (time-sliced rotation)
    return rr_pick(candidates)                            # e.g. front of RR queue

  # 3) Normal (OTHER/BATCH): CFS-ish fairness by smallest vruntime
  if exists t in runnable_threads where t.policy in {OTHER, BATCH}:
    return argmin_vruntime(runnable_threads where policy in {OTHER, BATCH})

  # 4) IDLE: only if nothing else runnable
  if exists t in runnable_threads where t.policy == IDLE:
    return any(runnable_threads where policy == IDLE)

  return idle_task()


# ----------------------------
# Events: timer tick & wakeup
# ----------------------------

function on_timer_tick(current, tick):
  if current.policy == RR:
    current.rr_slice_left -= tick
    if current.rr_slice_left <= 0:
      rr_enqueue_back(current)   # same RT priority queue
      reschedule()

  if current.policy in {OTHER, BATCH}:
    current.runtime  += tick
    current.vruntime += tick * weight_from_nice(current.nice)

    if should_preempt_for_fairness(current):
      reschedule()

  if current.policy == DEADLINE:
    # deadline enforcement not shown (budget/deadline checks)
    if deadline_violation_or_budget_exhausted(current):
      reschedule()


function on_thread_wakeup(woken, current):
  woken.state = RUNNABLE
  enqueue(woken)

  # RT preemption: higher RT priority immediately preempts normal/low RT
  if woken.policy in {FIFO, RR} and is_higher_rt_priority(woken, current):
    preempt_now()
    return

  # Normal wakeup preemption (latency heuristics)
  if woken.policy in {OTHER, BATCH} and current.policy in {OTHER, BATCH}:
    if wakeup_should_preempt_for_latency(woken, current):
      maybe_preempt()


# ----------------------------
# Main scheduling loop (per CPU)
# ----------------------------

function scheduler_loop():
  while true:
    runnable = all_threads where state == RUNNABLE
    next = pick_next(runnable)
    context_switch_to(next)
```

## Issues

- [std::execution::par_unseq](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) parallel execution policy (Usually a TBB ) could be triggering thread priority issues.

```cpp
        std::for_each(measurement.lidar_full_cloud_->points.begin(), measurement.lidar_full_cloud_->points.end(),
                      [&](const auto &pt) {
                          ASSERT_GE(pt.time, 0.0)
                              << "lidar point timestamp should be greater than or equal to zero";
                          ASSERT_LT(pt.time, lidar_timestamp)
                              << "lidar point timestamp should be earlier than lidar timestamp";
                      });
    };
```

**:** The test used [std::execution::par_unseq](vscode-file://vscode-app/usr/share/code/resources/app/out/vs/code/electron-browser/workbench/workbench.html) (parallel execution policy) which relies on TBB (Threading Building Blocks). TBB attempted to set thread priorities, but the container doesn't have real-time scheduling permissions (`ulimit -r` = 0).

**Fix:** change the parallel policy to `std::execution::seq`
