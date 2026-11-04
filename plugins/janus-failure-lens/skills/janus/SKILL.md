---
name: janus
description: Ask what the Janus failure classifier said about a failed tool call, and steer by its verdict.
---

# Janus

Janus is the Llama RLCD fine-tune that reads every failed tool call and
answers one question: debug, fix_input, or fix_call — with confidence.

After any failure, its one-line verdict is already in context:

```
[janus] debug (0.99) -- Read the error and change the code.
```

Steer by it: `debug` means the code is broken, `fix_input` means the
path or handle is wrong, `fix_call` means the call shape is wrong.
Below threshold the hook says it is unsure instead of naming a class —
check the failure before repeating it.
