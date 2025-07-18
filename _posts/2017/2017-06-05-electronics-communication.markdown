---
layout: post
title: Electronics - Communication Protocols
subtitle: Encoding / Decoding, Error Checking (CRC), Base64, EtherCAT
date: '2017-06-01 13:19'
header-img: "img/bg-material.jpg"
tags:
    - Electronics
---

## Error Checking

### CRC

#### GF(2) Math

`GF(2)` is pronounced as "Galois Field 2". Often used in cryptography, it's a binary field that contains only 0 and 1 for error correction. 

Some properties of `GF(2)`: 

- Subtraction is actually XOR here.
- Addition is XOR too. 
- Multiplication is and
- Division?

```
0 + 0 = 0
1 + 0 = 1
0 + 1 = 1
1 + 1 = 0

0 - 0 = 0
1 - 0 = 1
0 - 1 = 1
1 - 1 = 0

0 * 0 = 0
1 * 0 = 0
0 * 1 = 0
1 * 1 = 1
```

When we have larger numbers, we use an abstract placeholder variable `x` to form a polynomial. For example:

$$
\begin{gather*}
\begin{aligned}
& 1011 \rightarrow x^3 + x + 1
\end{aligned}
\end{gather*}
$$

And we apply the same addition subtraction rules (which is `XOR`):

$$
\begin{gather*}
\begin{aligned}
& 1011 \oplus 0110 =  x^3 + x + 1 + x^2 + x = x^3 + x^2 + 1
\\
& 1011 \ominus 0110 =  x^3 + x + 1 - x^2 - x = x^3 + x^2 + 1
\end{aligned}
\end{gather*}
$$

For multiplication, we need: $x^ax^b = x^{a+b}$. and reduce coefficients mod 2 (1+1=0) So we have:

$$
\begin{gather*}
\begin{aligned}
& 1011 \times 0110 = (x^3 + x + 1)(x^2 + x) 
\\ &
= x^5 + x^3 + x^2 + x^4 + x^2 + x = 
\\ &
= x^5 + x^4 + x^3 + x
\end{aligned}
\end{gather*}
$$

For division:
- Align the divisor with the highest degree of the dividend
- Subtract the two
- Shift the divident to the right, subtract, until the degree of the remainder is less than that of the divisor.

For example

$$
\begin{gather*}
\begin{aligned}
& (x^4 + x^3 + x^2 + 1)/(x^3 + x^2 +1) = x + y + remainder
\\ &
\rightarrow (x^4 + x^3 + x^2 + 1) - (x^3 + x^2 +1)x = x^2 + x + 1
\\ &
\text{Since: }
\\ & (x^2 + x + 1) = 0 \times (x^3 + x^2 +1) + (x^2 + x + 1)
\\ &
\text{Therefore: }
\\ & \rightarrow y = 0, remainder = x^2 + x + 1
\end{aligned}
\end{gather*}
$$

### CRC Math

CRC (Cyclic redundancy check) is a small value calculated on a piece of data to ensure data integrity. It was first proposed in 1961. Here, we walk through a simple CRC-3 example:

1. Choose our generator polynomial (the Divisor above) to be 3rd order polynomial (since it's CRC **-3**): $G(x) = x^3 + x + 1 (1011)$
2. Append input data `1101` by the highest degree of the generator (which is 3): `1101 -> 1101000`
3. Divide the appended input by the generator polynomial:

$$
\begin{gather*}
\begin{aligned}
& (1101000)/(1011) = 1100 + 100
\end{aligned}
\end{gather*}
$$

So the remainder is `100`, we choose that as the "checksum" of this message. We send the message with this checksum, then on the receiver end, we calculate the checksum of the received message, and compare that checksum with this checksum.

In reality, it's common to use CRC-16 or CRC-32.

## Encoding & Decoding

### Base 64

Base 64 is to encode binary data into a text-safe format. Email, JSON, HTTP, XML are all designed to work with text. This is great for avoiding issues caused by

- Line breaks such as  `\r`, `\n`. Such delimeters could prematurely terminate transmission, e.g., JSON.
    - File systems / databases may reserve `/` for directories, NULL. So really we can run into misinterpretation issues here.
- Special characters `<`, `>`, could be misinterpreted as commands, codes, etc. E.g., in XML, `&`, `<` must be escaped because they interfere with XML tags

Base 64 replaces binary data with with a safe standardized char set: `a-z`, `A-Z`, `0-9`, `+-/`.

So a sample workflow is:

```
Binary file (e.g., an image) is encoded as Base64. -> Transmitted over a protocol like email or API -> Decoded back into the original binary format by the receiver.
```

## EtherCAT

![](https://i.postimg.cc/SN7dDqsn/4kXDUB.gif)

Principle: the EtherCAT protocol works like a train. In the vanilla loop topology as shown, 
1. The master assembles data for each slave device into a single message, like a train
2. The first slave receives it at Ethernet port 1, and:
    - Copies the data matching its address, and 
    - Updates: it writes its own input/process‑data (sensor readings, status bits, etc.) into the slot of the frame that is reserved for it.
    - Increments work counter
3. Passes the message out from Ethernet port 2 to slave 2
4. Repeat the same process

In the OSI model, this is a level 3 (network) protocol:
- Physical
- Datalink 
- Network

Implementation:

- Additionally, there is a "Distributed Clock" (DC) mode. In that mode, the first DC device injects timestamps of message reception and departure. This helps the master to calculate "network delay", and keep everyone phased locked. 
- In the EtherCAT protocol, the main segments are still following the CAN protocol
- In total, 65536 devices are supported. 
- The master logic is software only
- The whole operation takes about 300ns, while TCP/IP takes 20ms. 

