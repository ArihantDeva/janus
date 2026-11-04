"""llama.cpp scoring backend: the wire Muse Code actually talks to.

The transformers path (train_lora.score) is convenient for evaluation but wrong
for serving: it holds the whole model in a Python process and cannot be reached
over HTTP. Muse Code's `--provider local --base-url <URL>` speaks an
OpenAI-shaped chat-completions API, so the guard has to be a server.

The key idea that keeps this honest: we do NOT ask the model to generate an
answer and parse it. We send the prompt, take ONE token of output, and read the
probability that the sampled token was the FAIL word. With `logprobs` enabled
and `top_logprobs` covering both candidates, the two log-probabilities give
p(fail) directly -- no generation, no parsing, no retry loop.

That is also why `max_tokens=1` is a correctness requirement, not an
optimisation: asking for more would make this autoregressive and would change
what is being measured.

This module is transport-only and takes an injectable `post`, so every test runs
without a server.
"""
import json
import math
import urllib.request


class LlamaCppScorer:
    """Score one command via a running llama-server.

    `post(url, payload) -> dict` is injected so tests need no network. The
    default posts JSON to the given URL.
    """

    def __init__(self, base_url="http://127.0.0.1:8080", timeout=30, post=None,
                 fail_word="no"):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._post = post or self._http_post
        self.fail_word = fail_word

    def _http_post(self, url, payload):
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def p_fail(self, prompt):
        """Probability that the model's next word is the FAIL word.

        Returns a float in [0, 1]. Raises RuntimeError if the server did not
        return the two candidate log-probabilities, because silently falling
        back to 0.5 would make the guard look calibrated while carrying no
        information.
        """
        payload = {
            "prompt": prompt,
            "max_tokens": 1,          # correctness: one token, never generate
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
            "echo": False,
        }
        out = self._post(f"{self.base_url}/v1/completions", payload)
        try:
            top = out["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                "llama-server did not return top_logprobs; the guard cannot "
                f"derive a probability without it ({e})") from e
        return self._pair_to_p(top, self.fail_word)

    @staticmethod
    def _pair_to_p(top, fail_word):
        """Two-way softmax over the FAIL and OK tokens from log-probs."""
        def lp(word):
            for k, v in top.items():
                if k.strip().lower() == word:
                    return float(v)
            return None

        lf = lp(fail_word)
        lo = lp("yes" if fail_word == "no" else "no")
        if lf is None or lo is None:
            raise RuntimeError(
                f"both answer tokens must appear in top_logprobs; got {list(top)}")
        # log-probs are already log-softmax over the full vocab, so the ratio is
        # the two-way conditional probability with no extra normalisation.
        # math.exp overflows for extreme log-prob gaps; the logistic must
        # saturate to 0 or 1 instead of raising, because a guard that crashes on
        # a confident model is worse than one that answers "certain".
        d = lo - lf
        if d > 700.0:
            return 0.0
        if d < -700.0:
            return 1.0
        return 1.0 / (1.0 + math.exp(d))
