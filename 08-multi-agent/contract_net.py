"""
Lab 2: The Contract Net: announce, bid, award
=============================================

The Contract Net Protocol that Reid G. Smith published in 1980 answers the question "who should do this job?":
  1. Announce: the manager broadcasts the task, saying which skill it needs and what the eligibility threshold is
  2. Bid: each contractor assesses for itself "can I do it, and how long will it take?", then bids if it wants to, or declines
  3. Award: the manager compares all the bids and gives the contract to the best one
  4. Report: the contractor hands back the result when it's done
FIPA later turned it into a standard, with the message names cfp / propose / refuse / accept-proposal / inform.

The key point: the manager **doesn't need to know** whether each contractor is busy or what it's good at; the contractors say so themselves in their bids.
This lab also shows what to do when nobody bids, and how much worse "taking turns, ignoring skills" does by comparison.

Run:
  python contract_net.py                     # mock model: contractors bid by rules based on their skill tables
  python contract_net.py --provider claude   # real model: the model writes every bid (calls the API dozens of times)
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import MockLLM, add_provider_args, chat, user_message  # noqa: E402

# Contractors: skill levels (0–1) and price per hour. The higher the level, the faster and better the work.
CONTRACTORS = [
    {"name": "Researcher", "skills": {"research": 0.9, "writing": 0.8, "summarizing": 0.7}, "price": 3},
    {"name": "Programmer", "skills": {"coding": 0.9, "debugging": 0.85, "writing": 0.4}, "price": 3},
    {"name": "Mathematician", "skills": {"math": 0.95, "coding": 0.6}, "price": 2},
    {"name": "Stenographer", "skills": {"summarizing": 0.8, "translation": 0.85, "writing": 0.6}, "price": 1},
]
# Tasks: (ID, description, skill needed, workload: hours needed by someone with a level of 1)
TASKS = [
    ("T1", "Research multi-agent frameworks since 2023", "research", 4),
    ("T2", "Implement a function that parses JSON logs", "coding", 3),
    ("T3", "Derive the computational complexity of attention", "math", 2),
    ("T4", "Translate the research findings into Chinese", "translation", 2),
    ("T5", "Fix a failing unit test", "debugging", 2),
    ("T6", "Write a 300-word project introduction", "writing", 2),
    ("T7", "Implement a BPE tokenizer", "coding", 4),
    ("T8", "Write a one-paragraph summary for each of 10 papers", "summarizing", 3),
    ("T9", "Draw a system architecture diagram", "drawing", 2),
]
THRESHOLD = 0.6            # the eligibility threshold in the announcement: the minimum level for this skill
RELAXED = 0.3              # what the second announcement relaxes it to when nobody bids
SKILLS = sorted({s for c in CONTRACTORS for s in c["skills"]})


# ---------------------------------------------------------------- contractors: assess themselves, bid themselves
def rule_bid(contractor, task, threshold, busy_until):
    """The mock model's bidding logic: bid only if it has the skill at or above the threshold; duration = workload / level."""
    _, _, skill, size = task
    level = contractor["skills"].get(skill, 0)
    if level < threshold:
        return {"bid": False, "reason": f"my {skill} level is only {level}, below the {threshold} threshold"}
    hours = round(size / level, 1)
    return {"bid": True, "hours": hours, "reason": f"{skill} level {level}; current work done by hour {busy_until:.1f}"}


def ask_contractor(contractor, task, threshold, busy_until, provider, model):
    """Send the task announcement to one contractor and get its bid back. In real mode, the model writes the bid."""
    tid, desc, skill, size = task
    prompt = (f"You are the \"{contractor['name']}\" on the team. Your skill levels are {json.dumps(contractor['skills'])}, "
              f"and your current work won't be finished until hour {busy_until:.1f}.\n"
              f"Task announcement {tid}: {desc}. Skill needed: {skill}; eligibility: a {skill} level of at least {threshold}; "
              f"workload: {size} hours for someone with a level of 1.\n"
              'Assess yourself honestly. Output only one JSON object: '
              '{"bid": true or false, "hours": the number of hours you expect to need, "reason": "a one-sentence reason"}')
    mock = MockLLM([lambda m, t: json.dumps(rule_bid(contractor, task, threshold, busy_until))])
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    try:
        bid = json.loads(re.search(r"\{.*\}", reply.text, re.S).group(0))
    except (AttributeError, json.JSONDecodeError):
        return {"bid": False, "reason": "(the bid was in the wrong format; treated as a decline)"}
    if bid.get("bid") and not isinstance(bid.get("hours"), (int, float)):
        return {"bid": False, "reason": "(the bid didn't give a clear duration; treated as a decline)"}
    return bid


# ---------------------------------------------------------------- the manager: announce, compare bids, award the contract
def reformulate(task, provider, model):
    """When nobody bids even with a relaxed threshold, the manager rephrases the task into a form someone on the team can do."""
    tid, desc, skill, size = task
    prompt = (f"Task {tid} \"{desc}\" needs the skill \"{skill}\", but nobody on the team bid on it. "
              f"The skills the team members have are: {', '.join(SKILLS)}.\n"
              "Rewrite this task into a form the team can complete. Output only JSON: "
              '{"description": "the rewritten task", "skill": "one skill from the list above"}')
    canned = {"description": "Write a text description of the system architecture diagram in Mermaid syntax "
                             "(it can be rendered into a diagram automatically)", "skill": "coding"}
    mock = MockLLM([json.dumps(canned)])
    reply = chat([user_message(prompt)], provider=provider, model=model, mock=mock, effort="low")
    try:
        new = json.loads(re.search(r"\{.*\}", reply.text, re.S).group(0))
        if new.get("skill") in SKILLS:
            return (tid + "'", new["description"], new["skill"], size)
    except (AttributeError, json.JSONDecodeError):
        pass
    return None


def contract_net(provider, model, verbose=True):
    busy = {c["name"]: 0.0 for c in CONTRACTORS}       # the hour at which each contractor's current work is done
    schedule = []                                       # (contractor, task ID, start, finish, level)
    say = print if verbose else (lambda *a, **k: None)

    for task in TASKS:
        attempts = [(task, THRESHOLD), (task, RELAXED)]
        while attempts:
            current, threshold = attempts.pop(0)
            tid, desc, skill, size = current
            say(f"\n📢 Announcement {tid}: {desc} (needs \"{skill}\", threshold {threshold})")
            bids, refused = [], []
            for c in CONTRACTORS:
                b = ask_contractor(c, current, threshold, busy[c["name"]], provider, model)
                if b.get("bid"):
                    finish = busy[c["name"]] + b["hours"]
                    bids.append((finish, b["hours"] * c["price"], c, b))
                    say(f"   📝 {c['name']} bids: {b['hours']} hours, delivers at hour {finish:.1f}, "
                        f"price {b['hours'] * c['price']:.1f} ({b['reason']})")
                else:
                    refused.append(c["name"])
            if refused:
                say(f"   ✋ Declined: {', '.join(refused)}")
            if bids:
                # the manager's rule for comparing bids: earliest delivery wins; on a tie, the cheaper one
                finish, cost, winner, b = min(bids, key=lambda x: (x[0], x[1]))
                start = busy[winner["name"]]
                busy[winner["name"]] = finish
                schedule.append((winner["name"], tid, start, finish, winner["skills"].get(skill, 0), cost))
                say(f"   ✅ Awarded to: {winner['name']} (hours {start:.1f}–{finish:.1f})")
                break
            say("   ⚠️  Nobody bid.")
            if not attempts and threshold == RELAXED:
                new = reformulate(current, provider, model)
                if new:
                    say(f"   🔁 The manager rewrote the task as: {new[1]}")
                    attempts.append((new, THRESHOLD))
                else:
                    say("   🙋 Still nobody can do it; escalating to a human.")
            elif attempts:
                say(f"   🔁 Relaxing the threshold and announcing again.")
    return schedule


# ---------------------------------------------------------------- control group: take turns, ignoring skills
def round_robin():
    busy = {c["name"]: 0.0 for c in CONTRACTORS}
    schedule = []
    for i, (tid, desc, skill, size) in enumerate(TASKS):
        c = CONTRACTORS[i % len(CONTRACTORS)]
        level = c["skills"].get(skill, 0.1)            # a job it can't do at all still has to be slogged through, slowly
        hours = round(size / level, 1)
        start = busy[c["name"]]
        busy[c["name"]] = start + hours
        schedule.append((c["name"], tid, start, start + hours, level, hours * c["price"]))
    return schedule


def gantt(schedule, scale):
    """A Gantt chart drawn with characters: each character stands for 1/scale of an hour."""
    end = max(s[3] for s in schedule)
    step = max(1, 4 // scale)                           # how many hours between tick labels, so each label takes 4 characters
    width = max(len(c["name"]) for c in CONTRACTORS) + 2
    print("   " + " " * width + "".join(f"{h:<4}" for h in range(0, int(end) + step, step)))
    for c in CONTRACTORS:
        row = [" "] * (int(end * scale) + 2)
        for name, tid, start, finish, *_ in schedule:
            if name != c["name"]:
                continue
            a, b = int(round(start * scale)), max(int(round(finish * scale)), int(round(start * scale)) + 1)
            label = (tid + "·" * (b - a))[: b - a]
            row[a:b] = list(label)
            row[a] = "["
        print("   " + c["name"].ljust(width) + "".join(row))


def report(name, schedule):
    makespan = max(s[3] for s in schedule)
    quality = sum(s[4] for s in schedule) / len(schedule)
    weak = sum(1 for s in schedule if s[4] < THRESHOLD)
    cost = sum(s[5] for s in schedule)
    print(f"   {name}| all done after {makespan:5.1f} hours | average level {quality:.2f} | "
          f"tasks given to non-experts: {weak} | total cost {cost:5.1f}")


def main():
    parser = add_provider_args(argparse.ArgumentParser(description="The Contract Net Protocol"))
    args = parser.parse_args()
    print(f"Team: {', '.join(c['name'] for c in CONTRACTORS)}; {len(TASKS)} tasks. "
          "The manager announces the tasks one at a time.")
    cn = contract_net(args.provider, args.model)
    if not cn:
        print("\nNot a single task was awarded. With a real model, first check whether it output only a JSON bid as asked.")
        return

    print("\nThe Contract Net's schedule ([1 means T1 starts, · means in progress; each cell is half an hour, "
          "and the scale is in hours):")
    gantt(cn, scale=2)
    rr = round_robin()
    print("\nControl group: the schedule when tasks are handed out in turn, ignoring skills "
          "(note: in this chart each cell is one hour):")
    gantt(rr, scale=1)
    print()
    report("Contract Net ", cn)
    report("Round robin  ", rr)
    print("""
Reading the results (based on the mock model's script):
  * From start to finish, the manager never asked "who can do what"; it only looked at the bids.
    The contractors know themselves best, so they are the ones who estimate the duration.
  * Load balancing appears on its own: after the Programmer took on two jobs, its bid for the T7 coding task
    could only deliver later, so the Mathematician, who can code and had free time, won the contract.
  * T9: nobody on the team can draw. Even with a relaxed threshold nobody bid, so the manager rewrote the task
    into a form the team can do; if nobody can be found at all, it escalates to a human.
  * Taking turns looks "fair," but it gave debugging (T5) and drawing (T9) to the Researcher and writing (T6)
    to the Programmer. The Researcher alone has to work more than 40 hours while everyone else sits idle early on:
    slow and expensive.

With large language models there's one more thing to think about: a bid written by a model **isn't necessarily honest**;
it may overestimate its own abilities. The protocol itself doesn't check whether a bid is true, so in a real system
the manager also has to look at each contractor's track record, or check the work on delivery.

Think about it:
  1. If you change the rule to "the cheapest bid wins," how do the schedule and the total cost change?
  2. What happens if one contractor always quotes half the time it really needs? How could the manager find out?
  3. Right now the tasks are announced one at a time. If all 9 were announced at once, the same contractor
     could win several contracts at the same time. What should happen then?""")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        raise SystemExit(f"Error: {e}")
