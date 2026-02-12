---
layout: post
title: Linux - Networking 2 UDP Traffic Profiling
date: 2018-02-02 13:19
subtitle: nftables
comments: true
tags:
  - Linux
---
## Loopback Traffic

When profiling communication-heavy systems (e.g., DDS / ROS 2 workloads), it’s common to look at loopback (`lo`) traffic to understand inter-process communication on a single host.

```bash
ip -s link show lo

1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT group default qlen 1000  
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00  
    RX:  bytes packets errors dropped  missed   mcast             
    4137004980 2508770      0       0       0       0   
    TX:  bytes packets errors dropped carrier collsns             
    4137004980 2508770      0       0       0       0
```

This result is the counters of bytes and packets  of all traffic traversing the loopback interface, including:

- UDP
- TCP
- ICMP

Note: SHM Is NOT Included because it does **not** traverse a network interface. Therefore, if your middleware (e.g., Cyclone DDS, Fast DDS) uses SHM transport, that traffic is invisible to `lo`.

Since `ip -s link` shows cumulative counters since boot, you need to separate out UDP specific traffic. To inspect UDP-specific kernel counters:

```bash
nstat -az | grep '^Udp'

UdpInDatagrams                  38376751991
UdpNoPorts                      173316502
UdpInErrors                     20263952
UdpOutDatagrams                 1734575680
UdpRcvbufErrors                 20263952
UdpSndbufErrors                 32182
UdpInCsumErrors                 0
UdpIgnoredMulti                 648380
```

Key fields:

|Counter|Meaning|
|---|---|
|`UdpInDatagrams`|UDP packets delivered to user space|
|`UdpOutDatagrams`|UDP packets sent by applications|
|`UdpNoPorts`|Packets received for a port with no listener|
|`UdpInErrors`|Total receive errors|
|`UdpRcvbufErrors`|Packets dropped due to full receive buffer|
|`UdpSndbufErrors`|Send buffer failures|

Note, `UdpRcvbufErrors == UdpInErrors` in your output, meaning most UDP errors are due to receive buffer exhaustion. That’s a strong signal of backpressure or undersized buffers.

---

## Counting UDP with iptables

### What is nftables / `nft`?

**nftables** is the modern Linux packet filtering and classification framework (the successor to iptables). It’s built into the kernel and configured using the **`nft`** command. Beyond firewalling, nftables is great for **instrumentation** because rules can carry **counters** (bytes/packets). That means you can attach a counter to “UDP packets on `lo`” and read back exact traffic volumes without needing `tcpdump, pcap parsing`, or application changes.

In this section, we’ll use nftables counters as a lightweight “UDP loopback meter.”

### 1) Install tools

```bash
sudo apt update  
sudo apt install -y nftables jq
```

**Why jq?**  `nft` can output structured JSON (`nft -j ...`), and jq makes it trivial to extract the byte counters reliably.

### 2) Create a fresh nftables table + chains

```bash
sudo nft delete table inet lo_prof 2>/dev/null || true  
sudo nft add table inet lo_prof  
  
sudo nft add chain inet lo_prof input  '{ type filter hook input priority 0; policy accept; }'  
sudo nft add chain inet lo_prof output '{ type filter hook output priority 0; policy accept; }'
```

**What this does:**

- Creates a dedicated table named `lo_prof` so your profiling rules are isolated and easy to remove later.

- Uses the **`inet`** family, which covers both IPv4 and IPv6 in one table (convenient and usually what you want on modern systems). DDS usually uses both

**About the chains:**

- The `input` chain is attached (“hooked”) to the kernel’s **INPUT path**, which sees packets destined for local sockets.

- The `output` chain is attached to the **OUTPUT path**, which sees locally generated packets leaving the host stack.

**Important detail:**  

- The chains are created with `policy accept`, so this setup is not intended to filter traffic—only to count it.

### 3) Add UDP loopback counter rules

```bash
sudo nft add rule inet lo_prof input  iifname "lo" meta l4proto udp counter accept  
sudo nft add rule inet lo_prof output oifname "lo" meta l4proto udp counter accept

Reset counters to start from zero:

sudo nft reset counters

```

**What this does:**

- Adds **two rules**, one per direction:

  - **INPUT**: count UDP packets whose **incoming interface** is `lo`

  - **OUTPUT**: count UDP packets whose **outgoing interface** is `lo`

- The `counter` expression increments packet+byte counters automatically.

- The rule ends with `accept` to ensure it doesn’t block or change behavior.

### 4) Verify the rules (and counters exist)

```bash
sudo nft list table inet lo_prof
```

You should see the two rules with `counter packets bytes`. **What to look for:**

- Two chains: `input` and `output`
- One rule in each chain that includes `counter packets bytes`

This is a sanity check that:

- The hooks are installed
- The rules match `lo` + UDP
- Counters are present

### 5) Read the counters once (input_bytes, output_bytes)

```bash
in_bytes=$(sudo nft -j list chain inet lo_prof input \  
  | jq -r 'first(.nftables[] | select(.rule?) | .rule.expr[]? | select(.counter?) | .counter.bytes) // 0')  
  
out_bytes=$(sudo nft -j list chain inet lo_prof output \  
  | jq -r 'first(.nftables[] | select(.rule?) | .rule.expr[]? | select(.counter?) | .counter.bytes) // 0')  
  
echo "input_bytes=$in_bytes output_bytes=$out_bytes"
```

**What this does:**

- Queries each chain in JSON form.
- Extracts the **byte** value from the first counter it finds.
- Prints a simple snapshot of cumulative bytes in each direction.

**Interpretation:**

- `input_bytes` = UDP bytes received via loopback (delivered toward local sockets)
- `output_bytes` = UDP bytes sent via loopback (originating from local apps)

On loopback, RX/TX often track closely, but they don’t have to match perfectly depending on what you’re counting and where drops occur.

### 6) Sample to CSV every second

```bash
echo "timestamp,input_bytes,output_bytes" > udp_lo.csv  
  
while true; do  
  ts=$(date +%s)  
  
  in_bytes=$(sudo nft -j list chain inet lo_prof input \  
    | jq -r 'first(.nftables[] | select(.rule?) | .rule.expr[]? | select(.counter?) | .counter.bytes) // 0')  
  
  out_bytes=$(sudo nft -j list chain inet lo_prof output \  
    | jq -r 'first(.nftables[] | select(.rule?) | .rule.expr[]? | select(.counter?) | .counter.bytes) // 0')  
  
  echo "$ts,$in_bytes,$out_bytes" >> udp_lo.csv  
  sleep 1  
done

# Stop with **Ctrl+C**.
```

**What this does:**

- Creates a time-series CSV of counters.
- Every second, it records:
  - Unix timestamp
  - cumulative input bytes
  - cumulative output bytes

**Why log cumulative counters (not bytes/sec)?**

- Cumulative counters are robust and simple.
- You can compute rates later by taking deltas between consecutive rows:
  - `bytes/sec ≈ (bytes[t] - bytes[t-1]) / (ts[t] - ts[t-1])`

This makes post-processing easy (Python, pandas, gnuplot, etc.).
