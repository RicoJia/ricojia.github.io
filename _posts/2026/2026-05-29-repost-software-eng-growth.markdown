---
layout: post
title: "[Career Advice] REPOST - What 16 years of software engineering taught me about growth"
date: 2026-05-29 13:19
subtitle: ""
comments: true
header-img: img/post-bg-infinity.jpg
tags:
  - Career Advice
---

This article [was originally posted here](https://leaddev.com/career-development/what-16-years-of-software-engineering-taught-me-about-growth)

In service companies, ignoring documentation meant chaotic client handoffs. In product companies, skipping community meant slower growth. In my startup, avoiding experimentation meant stagnation.

Documentation is not an afterthought. A well-written README file, **inline comments explaining the “why” behind a decision rather than just the “what,”** **and decision records help your teammates understand the reasoning for implementation choices**. When my team [migrated](https://leaddev.com/technical-direction/migration-trap-software-engineering) a 12-year-old Java codebase to Kotlin in under a year, the engineers who had documented their technical decisions and tradeoffs, even informally, moved significantly faster than those who hadn’t. **You will forget why you made a choice six months from now**. Writing doesn’t always have to be public; it can be **a personal journal of your technical decisions, questions, tradeoffs, and learnings**. I keep mine in Apple Notes or a Google Doc. Sometimes you’ll be the only one who reads it. That’s fine. Your future self will thank you and those notes come in surprisingly handy during performance reviews.
- Discover the problem and user needs, define scope and constraints, develop the architecture with clear tradeoffs, and deliver through testing, release, and measurable outcomes.
- Start by being clear about what you’re trying to achieve before writing code because not every solution needs to be perfect. **Sometimes “good enough” is the right answer**. I learned this the hard way in my startup where over-engineering early features without validating usability cost us valuable time.
- whether that is readability, performance, or testability, so you can stay consistent and align your team around how and why decisions are made.

## Building community

I have watched talented engineers burn out trying to solve everything solo. The best features I’ve shipped were never the product of one person working in isolation – they came from collaboration, debate, and someone catching what I missed. When I was building my startup **, I realized quickly that without a community to pressure-test my ideas**, I was just running on personal conviction. That’s a dangerous way to make technical decisions. Community doesn’t have to mean a formal network. It can be your immediate team, an online forum, a Slack channel, or a local meetup. What matters is **having people around you who will challenge your thinking**, **share what they know**, and tell you **when you’re overcomplicating something**.

Code reviews are one of the most underused learning tools in engineering. Giving and receiving them isn’t just about catching bugs, it’s about understanding different approaches and tradeoffs. Some of my biggest technical learnings came from code review comments I’ve received.

Also, it’s equally valuable to talk through a problem before you’ve even started solving it. I’ve lost count of how many times explaining a complex issue to a colleague revealed the answer. I avoided code reviews early in my career because I was terrified people would discover I was just stack-overflowing everything. Turns out, everyone was doing the same thing.

Beyond your immediate team, the broader engineering community is more accessible than it appears. LinkedIn, Slack channels, **Reddit communities like r/androiddev**, and local or virtual meetups are all genuine entry points. **Contributing to open source libraries you use daily** is one of the fastest ways to understand how experienced engineers structure code at scale.

## Experimentation

In **service companies, experimentation was quietly discouraged**. **Billable hours left little room for curiosity**, **and time spent exploring was time not invoiced**. Product companies were different. **Innovation hubs and hackathon weeks created structured space** to try things without consequence. In my startup, there was no such structure. **Experimentation wasn’t a perk, it was survival**. Pivot or die is not a metaphor when you’re the one taking the final decision.

The most honest thing I can tell you about learning a new library or architecture pattern is that reading blogs or watching videos is not the same as **building with it**. You won’t **understand the tradeoffs until you’ve faced them yourself**. I **have a folder full of half-finished side projects that never shipped and never needed to**. They taught me more than most production work because **the stakes were low enough to actually break things intentionally.**

The **other thing experimentation builds is intuition**. The **more patterns you’ve tried**, the better your instinct becomes when choosing an approach under pressure. That instinct doesn’t come from documentation, **it comes from experience**. When I rebuilt the Wattpad comments experience using Jetpack Compose, it started as an internal experiment before it became a production feature. The only reason it worked was because I **took calculated risks of trying out new technology**. Staying current matters too, especially on mobile platforms. **UI libraries evolve constantly**. OS ships new features regularly. Engineers **who only learn through Jira tickets fall behind**. Engineers **who experiment stay sharp**.

A practical habit: **block one to two hours a week** for unstructured learning time. **No tickets, no meetings, just exploration**. I **try to do this on Friday afternoons**. If you want to try a new approach in production without the risk, use feature flags. **Test internally, iterate, and only ship when you’re confident**. It removes **the fear of experimentation from an environment where failure has real consequences**.

## Managing imposter syndrome

Every engineer I’ve worked with has felt it. The quiet fear of being found out, of not being good enough, of making the wrong call on an architecture decision that can’t easily be undone. It shows up differently depending on the environment.

In service companies, it was the fear of not knowing the client’s domain well enough. In product companies, it was the fear of making a call that would affect millions of users. **In my startup, it was the fear that every decision I made alone could sink the whole thing.** 

Imposter syndrome never fully disappears. John Ternus, Apple’s incoming CEO, admitted in a 2024 commencement speech at the University of Pennsylvania that, on his first day at Apple, he wasn’t sure he belonged – that **everyone around him seemed smarter and more confident**. Someone with 25 years at Apple and a CEO title ahead of him felt it too. That should tell you something. 

I’ve felt it at every company I’ve worked at, and I’ve seen it in engineers far more experienced than me. What changes is **your relationship with it**. The fear of breaking things in production is real and worth respecting but it shouldn’t freeze you. **Production incidents happen to everyone**. What matters is how you respond: **transparently, methodically, and without blame**. **Blameless postmortems** exist for a reason.

The most damaging form I’ve encountered is analysis paralysis, **leading to overthinking architecture decisions**. My startup years cured me of this. Perfect code that ships late is worthless. Timebox your decisions: **give yourself a deadline to research, prototype, and decide, then commit**. **You can iterate. You can’t recover lost time.**

**Two habits that actually helped me:** saying “I don’t know, but I’ll find out” out loud, without apology, and **deliberately celebrating wins like a tricky bug fixed, a messy class refactored, or a teammate unblocked**. These aren’t small things. Treating them as small is what feeds the imposter.

## Growing as a software engineer

I didn’t learn any of this cleanly or in order. **Documentation discipline came from painful handoffs in service companies**. Community came from the isolation of startup life. Experimentation came from product teams that gave us space to fail safely. Managing fear came from all three, repeatedly.

For engineers, these four practices compound over a career. For engineering leaders, the questions are slightly different: 

- Are you creating the conditions where your team can actually practice them?
- Does your team have **psychological safety to experiment**?
- Is **documentation celebrated or treated as overhead**?
- **Are code reviews learning opportunities or gatekeeping exercises?**

The engineers who **grow sustainably** aren’t necessarily the most technically gifted. They’re the ones **who stayed curious**, **built relationships**, **tried things outside their comfort zone**, and **kept moving despite the fear**.
