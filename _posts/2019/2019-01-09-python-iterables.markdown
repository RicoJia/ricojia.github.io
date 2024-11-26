---
layout: post
title: Python - Python Iterables
date: '2019-01-05 13:19'
subtitle: Iterable
comments: true
header-img: "img/post-bg-2015.jpg"
tags:
    - Python
---

## What is An Iterable

An iterable is any object that returns an iterator, which allows you to iterate its elements. Some common examples include lists, dictionaries, sets, and strings. Iterating over theses follows the "Iterable protocol".

    - Small exception: Dict is an iterable. But iter(di) gives you the keys

Iteratble Protocol:
    - Implements ```iter()``` to return an iterator.
        - Use `for i in iter(next(MyIterable))` to loop over
    - An iterator should have  `next()` to return the next object on its sequence. Raises `StopIteration` when there's no more items.

A baby example is:

```python
class BdayIterator:
    def __init__(self):
        self.count = 0

    def __next__(self):
        self.count += 1
        if self.count < 10:
            return 100
        else:
            raise StopIteration  # or do this

# __iter__ returns an iterator
bday = BdayIterator()

# manually iterating
while True:
    # __next__ returns the next value of the iterable
    try:
        num = next(bday)
    except StopIteration:
        print("Iteration stopped")
        break

class BdayIterable():
    def __iter__(self):
        return BdayIterator()
for i in BdayIterable():
    print("bday iterable: ", i)

ls = [1, 2, 3]
ls_iter = iter(ls)
# see 1, 2
print(next(ls_iter), next(ls_iter))

# create an iterable from dictionary
di = {"one": 1, "two": 2}
dict_iterator = iter(di)
print(next(dict_iterator))
```

Or we can also combine them, **which is a common practice**

```python
class MyIter:
    def __init__(self):
        self.count = 0
    def __next__(self):
        self.count += 1
        if self.count < 10:
            return self.count
        else:
            raise StopIteration  # or do this
    def __iter__(self):
        return self
it = MyIter()
for i in it:
    print(i)
```

## Generator And Itertools

A generator is a type of iterator that returns a value on the fly while it's called. So, everytime it will only load the current value into memory, which makes it memory efficient. One type of generator is a function that `yield` a value. So each time it's called, `next(generator)` (note `__iter__` is synthesized), the generator function pauses at `yield`. Here, we introduce some common methods in the `itertools` library.

### Islice

`islice` (yee-slice) means `iterator slice`. `itertools.islice(generator_func, start_id, end_id)` creates a slice `[start_id, end_id)`. Here we go one example:

```python
def test_slice_iterator():
    """
    """
    # 1
    from itertools import islice
    def count(n):
        while n < 200:
            yield n
            n += 1
    c = count(0)

    for i in islice(c, 10, 20):
        print(i)
```

2. `dropwhile(predicate, generator_func)`

`dropwhile()` will return a generator that drops elements from the beginning of the iteratble to the first element that makes `predicate` false

```python
# 2
from itertools import dropwhile
c = count(0)

data = [1, 3, 5, 2, 4, 6]
result = list(itertools.dropwhile(lambda x: x < 4, data))
print(result)  # Output: [5, 2, 4, 6]
```

- So the overarching difference from `filter()` is that `filter()`  returns all elements that makes `predicate` **`True`**, while `dropwhile()` discards elements until the first element that make `predicate` `False`.

### Permutations

- Saw now we'd like to generate all permutations of 3 words among a list of words

```python
import itertools
words = ['apple', 'banana', 'cherry', 'date']
permutations = list(itertools.permutations(indices, 3))
itertools.permutations(words, 3)
```

## List

- `for i in reversed(range(T_x)):`: reverse a list / iterable
- `ls = list(str)` Decompose a string into a list of letters
- `index = ls.index(element)`: find the index of an element
