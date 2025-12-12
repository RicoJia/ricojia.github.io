---
layout: post
title: C++ - [Concurrency 9] Scoped Lock and `adopt_lock`
date: '2023-06-01 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - C++
---

## `adopt_lock`

In a bank transfer system, if we transfer `A->B` and `B->A` concurrently. `std::lock()` locks two locks in a dead-lock free way, but does not unlock automatically, so we need to unlock them. One method is to transfer the mutex ownership to `lock_guard` with `std::adopt_lock`. `std::adopt_lock` is a tag saying "I've acquired the lock properly, just transfer the mutex ownership to me". Then, `lock_guard` will unlock.

```cpp
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

struct Account {
    explicit Account(int balance) : balance(balance) {}
    int balance;
    std::mutex m;
};

void transfer(Account& from, Account& to, int amount)
{
    // Lock both mutexes *atomically* to avoid deadlock
    std::lock(from.m, to.m);

    // These lock_guards adopt already-locked mutexes
    std::lock_guard<std::mutex> lock_from(from.m, std::adopt_lock);
    std::lock_guard<std::mutex> lock_to(to.m, std::adopt_lock);

    // ---- critical section: both accounts are locked here ----
    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance   += amount;
        std::cout << "Transferred " << amount
                  << " (from=" << &from << " to=" << &to << ")\n";
    } else {
        std::cout << "Insufficient funds (" << &from << ")\n";
    }
    // ---------------------------------------------------------
    // When this function returns, lock_from and lock_to go out of scope
    // and automatically call from.m.unlock() and to.m.unlock().
}

int main()
{
    Account a{1000};
    Account b{1000};

    // Two threads transferring in opposite directions.
    // Without std::lock(...) this could deadlock if they lock in opposite orders.
    auto t1 = std::thread([&] {
        for (int i = 0; i < 100; ++i) {
            transfer(a, b, 5);
        }
    });

    auto t2 = std::thread([&] {
        for (int i = 0; i < 100; ++i) {
            transfer(b, a, 5);
        }
    });

    t1.join();
    t2.join();

    std::cout << "Final balances: a=" << a.balance
              << ", b=" << b.balance << "\n";
}
```

## Scoped_Lock (C++17)

```cpp
void transfer(Account& from, Account& to, int amount)
{

    std::scoped_lock lock(from.m, to.m);

    // ---- critical section: both accounts are locked here ----
    if (from.balance >= amount) {
        from.balance -= amount;
        to.balance   += amount;
        std::cout << "Transferred " << amount
                  << " (from=" << &from << " to=" << &to << ")\n";
    } else {
        std::cout << "Insufficient funds (" << &from << ")\n";
    }
}

```
