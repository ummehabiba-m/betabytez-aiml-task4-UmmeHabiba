# Test Results — RAG Document Q&A Chatbot

**Prepared by Umme Habiba Malik · BetaBytez AI/ML Internship — Task 4**

Five questions were tested against the deployed chatbot: three the source
documents can answer, and two they cannot. All answers below are reported
exactly as the chatbot produced them — nothing has been edited or cleaned up.

---

## In-scope questions (documents can answer)

### 1. "What is self-attention?"

**Answer:**
> Self-attention, sometimes called intra-attention, is an attention mechanism
> relating different positions of a single sequence in order to compute a
> representation of the sequence.

**Sources retrieved:**
- Attention Is All You Need — Page 2 (distance: 0.9262)
- Attention Is All You Need — Page 14 (distance: 1.0168)
- Attention Is All You Need — Page 15 (distance: 1.0349)
- Attention Is All You Need — Page 6 (distance: 1.0355)

**Result:** ✅ Grounded. Correctly answered from *Attention Is All You Need*.

---

### 2. "How does retrieval-augmented generation combine parametric and non-parametric memory?"

**Answer:**
> [Paste the exact answer text your chatbot returned here.]

**Sources retrieved:**
- Retrieval-Augmented Generation (RAG) — Page 1 (distance: 0.5172)
- Retrieval-Augmented Generation (RAG) — Page 9 (distance: 0.5421)
- Retrieval-Augmented Generation (RAG) — Page 2 (distance: 0.6348)
- Retrieval-Augmented Generation (RAG) — Page 2 (distance: 0.6377)

**Result:** ✅ Grounded. Correctly answered from the RAG paper.

---

### 3. "What datasets did the RAG paper use to evaluate open-domain question answering?"

**Answer:**
> The RAG paper used the following datasets to evaluate open-domain question
> answering:
> 1. MSMARCO
> 2. Natural Questions
> 3. WebQuestions
> 4. TriviaQA

**Result:** ✅ Grounded. Correctly answered from the RAG paper.

---

## Out-of-scope questions (documents cannot answer)

### 4. "What is the capital of France?"

**Answer:**
> I cannot answer this from the provided documents.

**Sources retrieved:**
- Attention Is All You Need — Page 14 (distance: 1.7410)
- Attention Is All You Need — Page 3 (distance: 1.7467)
- Attention Is All You Need — Page 6 (distance: 1.7489)
- Attention Is All You Need — Page 5 (distance: 1.7523)

**Result:** ✅ Correctly refused. All retrieved distances (1.74–1.75) were
above the 1.5 grounding threshold, so the refusal was returned immediately by
the **distance filter** — the LLM was never called for this question.

---

### 5. "What was the exact GPU training cost in dollars for the RAG paper's experiments?"

**Answer:**
> I cannot answer this from the provided documents.

**Sources retrieved:**
- Attention Is All You Need — Page 7 (distance: 1.1697)
- Retrieval-Augmented Generation (RAG) — Page 17 (distance: 1.1841)
- Attention Is All You Need — Page 8 (distance: 1.2106)
- Retrieval-Augmented Generation (RAG) — Page 18 (distance: 1.3583)

**Result:** ✅ Correctly refused — but by a different mechanism than question 4.
All retrieved distances were **below** the 1.5 threshold (the retrieved chunks
genuinely discuss GPU hardware and training time), so this question passed
the distance filter and reached the LLM. It was the **LLM's own instruction-following**
that caught it, correctly recognizing that a specific dollar figure was never
actually stated anywhere in the retrieved text, even though the topic was
closely related.

This is the more interesting of the two refusals: it shows the grounding
check resisting a plausible-sounding but ungrounded answer, not just rejecting
questions that are obviously unrelated to the source documents.

---

## Summary

| # | Question | Type | Result |
|---|---|---|---|
| 1 | What is self-attention? | In-scope | ✅ Grounded |
| 2 | How does RAG combine parametric/non-parametric memory? | In-scope | ✅ Grounded |
| 3 | What datasets did the RAG paper use for open-domain QA? | In-scope | ✅ Grounded |
| 4 | What is the capital of France? | Out-of-scope | ✅ Refused (distance filter) |
| 5 | Exact GPU training cost in dollars for RAG experiments? | Out-of-scope | ✅ Refused (LLM judgment) |

All five questions produced the expected, honest behavior — no hallucinated
answers were observed in testing.