# Engineering & ML Mastery Guide

A living reference. Update it as you learn — cross out commands you've mastered, add ones you keep forgetting, tweak the workflows once you find what actually works for you.

---

## 1. LeetCode Problem-Solving Workflow

Run every problem through this sequence, out loud, even when alone. Interviewers are grading the process, not just the final code.

1. **Read twice, restate the problem in your own words.** Say the input, output, and constraints out loud before touching anything else.
2. **Nail down constraints.** Input size (n)? Value ranges? Duplicates allowed? Sorted? Negative numbers? This tells you the expected time complexity before you write a line of code (n ≤ 10 → exponential OK; n ≤ 10⁴ → O(n log n) or O(n²); n ≤ 10⁶+ → O(n) or O(n log n) only).
3. **Identify the pattern.** Ask: what data structure does this smell like? (See Section 5 catalog.) Say the pattern name out loud: "this looks like a sliding window problem because we want a contiguous subarray satisfying a condition."
4. **State a brute force first**, even if obviously too slow. This anchors your baseline complexity and often reveals the insight for the optimization.
5. **Talk through the optimized approach before coding.** Walk through it on a small example by hand. This is the step people skip and then get lost mid-implementation.
6. **Implement.** Narrate what you're writing as you write it, like you would in a real interview.
7. **Trace through with a test case**, including at least one edge case (empty input, single element, all duplicates, negative numbers).
8. **Analyze time and space complexity** — explicitly state both, and justify each (not just "it's O(n)" but "we visit each element once and do O(1) work per element").
9. **Reflect.** Was there a cleaner approach? Look at the editorial/discussion *after* solving, not before. Tag the pattern for spaced repetition (see tracker below).

> **Notes / adjustments to this workflow:**
>

---

## 2. ML / Textbook Study Workflow

1. **Read the section once for the big picture** — don't stop to fully absorb every equation on the first pass.
2. **Re-read and derive.** For any non-trivial equation, derive it by hand on paper rather than just reading the derivation. This is where retention actually happens.
3. **Implement it from scratch** where feasible (no library) — e.g., backprop for a 2-layer net, gradient descent, a small attention mechanism. Library implementations come after you've built the primitive yourself once.
4. **Connect to code you've already written.** Does this concept show up in a library function you use? Go read that source.
5. **Spaced repetition.** Turn key derivations, definitions, and gotchas into Anki cards same-day, while it's fresh.
6. **Weekly synthesis.** Once a week, explain (out loud or written, no notes) one concept you learned that week as if teaching it. Gaps show up immediately.

> **Notes / adjustments to this workflow:**
>

---

## 3. Command Line Reference

### Git
| Command | What it does |
|---|---|
| `git switch -c <branch>` | create + switch to new branch |
| `git restore --staged <file>` | unstage a file |
| `git restore <file>` | discard unstaged changes to a file |
| `git rebase -i HEAD~n` | interactive rebase last n commits |
| `git stash` / `git stash pop` | shelve and restore uncommitted changes |
| `git log --oneline --graph --all` | compact visual history |
| `git bisect start` | binary search for the commit that introduced a bug |
| `git reflog` | recover "lost" commits/branches |

> **Notes:**
>

### Docker
| Command | What it does |
|---|---|
| `docker ps -a` | list all containers, including stopped |
| `docker logs -f <container>` | follow logs |
| `docker exec -it <container> sh` | shell into a running container |
| `docker build -t name:tag .` | build image from Dockerfile |
| `docker system prune -a` | clean up unused images/containers/networks |
| `docker inspect <container>` | full JSON config/state of a container |
| `docker compose up -d --build` | rebuild and run in detached mode |

> **Notes:**
>

### grep / find / sed / awk
| Command | What it does |
|---|---|
| `grep -rn "pattern" .` | recursive search with line numbers |
| `grep -E "regex"` | extended regex |
| `grep -v "pattern"` | invert match (lines NOT matching) |
| `find . -name "*.py"` | find files by name pattern |
| `find . -mtime -1` | files modified in the last day |
| `find . -type f -exec grep -l "pattern" {} \;` | find files containing a pattern |
| `sed -i 's/old/new/g' file` | in-place find/replace |
| `sed -n '5,10p' file` | print lines 5–10 |
| `awk '{print $1}' file` | print first column |
| `awk -F, '{print $2}' file.csv` | print 2nd CSV column |

> **Notes:**
>

### kubectl / k9s
| Command | What it does |
|---|---|
| `kubectl get pods -A` | list pods, all namespaces |
| `kubectl logs -f <pod>` | follow pod logs |
| `kubectl exec -it <pod> -- sh` | shell into a pod |
| `kubectl describe pod <pod>` | detailed pod status/events (debugging) |
| `kubectl apply -f file.yaml` | apply a manifest |
| `kubectl rollout restart deployment <name>` | restart a deployment |
| `k9s` | terminal UI overview of cluster state |

> **Notes:**
>

### General shell
| Command | What it does |
|---|---|
| `command --help` / `man command` | first stop before asking AI |
| `history \| grep <term>` | search shell history |
| `<cmd> \| xargs <cmd2>` | pipe output as arguments to next command |
| `ctrl+r` | reverse search shell history interactively |
| `!!` | rerun last command |
| `<cmd> 2>&1 \| tee log.txt` | capture stdout+stderr to file and screen |

> **Notes:**
>

---

## 4. VS Code Keyboard-Only Reference

### Navigation (no mouse, no scrollbar)
| Shortcut | Action |
|---|---|
| `Cmd/Ctrl+P` | fuzzy-find file by name |
| `Cmd/Ctrl+Shift+O` | jump to symbol in current file |
| `Cmd/Ctrl+T` | jump to symbol across workspace |
| `F12` | go to definition |
| `Shift+F12` | find all references |
| `Alt+Left / Alt+Right` | back/forward through navigation history |
| `Cmd/Ctrl+Shift+E` | focus file explorer (then arrow keys) |
| `` Ctrl+` `` | toggle integrated terminal |

### Editing & multi-cursor
| Shortcut | Action |
|---|---|
| `Cmd/Ctrl+D` | select next occurrence of current selection |
| `Cmd/Ctrl+Shift+L` | select all occurrences at once |
| `Alt+Click` | add cursor at click point (fine, still keyboard-first workflow otherwise) |
| `Cmd/Ctrl+Shift+K` | delete line |
| `Alt+Up/Down` | move line up/down |
| `Shift+Alt+Up/Down` | copy line up/down |

### Refactoring & search
| Shortcut | Action |
|---|---|
| `F2` | rename symbol project-wide (scope-aware) |
| `Cmd/Ctrl+Shift+F` | search/replace across whole project |
| `Cmd/Ctrl+.` | quick fix / code actions |

### Vim motions (with Vim extension)
| Motion | Action |
|---|---|
| `ci"` / `ci(` | change inside quotes/parens |
| `dap` | delete a paragraph |
| `.` | repeat last edit |
| `daw` | delete a word |
| `>>` / `<<` | indent/outdent line |

> **Notes / shortcuts I keep forgetting:**
>

---

## 5. LeetCode Pattern Catalog (recognition cues)

| Signal in the problem | Likely pattern |
|---|---|
| Contiguous subarray/substring + condition | Sliding window |
| Sorted array, pair/triplet sum | Two pointers |
| "Kth largest/smallest", streaming data | Heap |
| Tree traversal, level-by-level | BFS |
| Explore all paths / backtracking needed | DFS + backtracking |
| Overlapping subproblems, optimal substructure | Dynamic programming |
| Graph reachability / shortest path | BFS (unweighted) / Dijkstra (weighted) |
| "Number of connected components" | Union-Find or DFS/BFS |
| Fixed-size range min/max as window slides | Monotonic deque |
| Intervals, merging/overlap | Sort + sweep |
| Prefix sums needed repeatedly | Prefix sum array / hashmap of running sums |
| Linked list, cycle or middle element | Fast & slow pointers |
| Top-K / frequency-based | Hashmap + heap |

> **Patterns I still mix up / need more reps on:**
>

---

## 6. Progress Tracker

| Date | LeetCode # | Pattern | Solved unaided? | Redo in (days) |
|---|---|---|---|---|
| | | | | |

| Date | Topic (textbook/course) | Derived by hand? | Implemented from scratch? |
|---|---|---|---|
| | | | |

---

## 7. Friction Ramp Check-in

Started: ___________  Hard deadline to reassess: ___________ (6 weeks out)

- [ ] Neovim motions feel automatic in LeetCode sessions
- [ ] CLI tools (git/docker/grep/kubectl) no longer require conscious lookup for common cases
- [ ] Still using `--help` before asking AI, by default
- [ ] Reviewing every AI-suggested diff line by line before accepting

If most of these are checked and output still feels slow, the bottleneck has shifted from tooling to content (LeetCode reps / ML fundamentals) — treat it as that, not a tooling problem.
