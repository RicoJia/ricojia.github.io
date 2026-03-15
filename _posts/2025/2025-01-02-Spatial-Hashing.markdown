---
layout: post
title: Robotics - Spatial Hashing
date: '2025-01-02 13:19'
subtitle: 
comments: true
header-img: "img/post-bg-unix-linux.jpg"
tags:
    - Robotics
---

## Spatial Hashing

Hash integer coordinates `<x,y,z>`. A good hash should make small input changes produce very different outputs.

```cpp
std::size_t operator()(const TileKey &k) const {
	// Combine with a simple hash
	std::size_t h = std::hash<int>()(k.x);
	h ^= std::hash<int>()(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
	h ^= std::hash<int>()(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
	return h;
}
```

This is essentially the classic `Boost hash_combine style formula`. It is widely used for combining hashes of multiple fields.:

```bash
seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
```

- `^=` XORs the new mixed value into the current hash.
    - `h ^= hash(y) + C + (h << 6) + (h >> 2);` is basically `h ^= (std::hash<int>()(k.y) + C + (h << 6) + (h >> 2));`

- `0x9e3779b9`: A magic constant often used in hash combining. It comes from the fractional part of the golden ratio and helps spread bits around.

- `(h << 6) and (h >> 2)` These shift the current hash left and right before mixing, so the order matters and nearby values do not combine too simply.