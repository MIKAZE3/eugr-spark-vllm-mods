import json, sys, time, urllib.request

key = open("/home/yztai/eugr-b12x-logs/vllm-api-key").read().strip()
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima".split()

def send(rid, nw):
    rng = __import__("random").Random(rid)
    txt = " ".join(rng.choice(words) + str(rng.randint(0, 99999)) for _ in range(nw))
    body = json.dumps({
        "model": "DeepSeek-V4-Flash-0731",
        "messages": [{"role": "user", "content": txt + " Reply with the single word: ok"}],
        "max_tokens": 64, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "http://192.168.88.11:8000/v1/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=600)
    resp = json.loads(r.read())
    dt = time.time() - t0
    msg = resp["choices"][0].get("message") or {}
    content = (msg.get("content") or msg.get("reasoning_content") or "")[:40]
    return dt, content

# pass 1: cold store (~3K tokens)
t_cold, c1 = send("replaychk", 3000)
print(f"[cold]    {t_cold:.1f}s content={c1!r}", flush=True)
time.sleep(2)
# pass 2: replay -> must exercise lookup/prepare_load/cuFileRead
t_rep, c2 = send("replaychk", 3000)
print(f"[replay]  {t_rep:.1f}s content={c2!r}", flush=True)
ok = "ok" in (c1 + c2).lower()
print(f"REPLAY-SMOKE {'PASS' if ok else 'CHECK'} speedup={t_cold/max(t_rep,0.01):.2f}x", flush=True)
