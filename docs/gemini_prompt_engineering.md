# Gemini Prompt Engineering Best Practices

Research compiled 2026-04-15 from official Google documentation and practitioner guides.

---

## 1. Official Google Prompt Design Guidelines

Source: [Prompt design strategies (ai.google.dev)](https://ai.google.dev/gemini-api/docs/prompting-strategies)

**Core principles:**
- Give clear, explicit instructions. Don't assume the model has context.
- Few-shot examples significantly outperform zero-shot. Start with 2-3 examples minimum.
- Ensure identical formatting across all few-shot examples (XML tags, whitespace, newlines).
- Place critical instructions first in the prompt.
- For long context inputs, put the data first, then the question at the end.
- Use bridging phrases like "Based on the information above..." after large data blocks.

**Gemini 3 specific (from [Gemini 3 Developer Guide](https://ai.google.dev/gemini-api/docs/gemini-3)):**
- Be concise and direct. Gemini 3 responds best to stripped-down instructions.
- Remove verbose chain-of-thought scaffolding from Gemini 2.x prompts.
- Keep temperature at default 1.0. Setting below 1.0 causes looping and degraded performance.
- Default output is terse/efficient. Explicitly request conversational tone if needed.
- Replace explicit chain-of-thought prompting with `thinking_level: "high"` parameter.

---

## 2. Structured Prompt Formatting: XML vs Markdown vs Plain Text

Source: [Prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies), [Gemini 3 Prompting Guide (philschmid.de)](https://www.philschmid.de/gemini-3-prompt-practices)

**Both XML tags and Markdown headers work well.** Google's official docs show examples of both. The key rule: **pick one and use it consistently within a single prompt. Never mix them.**

### XML approach (recommended for complex/structured prompts):
```xml
<role>
You are a procedural video analyst.
</role>

<constraints>
1. Only report events you are confident about.
2. Use step_id from the procedure.
</constraints>

<context>
{procedure steps and audio transcripts here}
</context>

<task>
Analyze this video frame and report any step completions or errors.
</task>
```

### Markdown approach (recommended for readability-first prompts):
```markdown
# Identity
You are a procedural video analyst.

# Constraints
- Only report events you are confident about.
- Use step_id from the procedure.

# Output format
Return a single JSON object.
```

**When to use which:**
- XML tags: Better for machine-readable separation, complex nested structures, agent workflows
- Markdown headers: Better for readability, simpler prompts, when you want the model to also output markdown
- Plain text: Still works fine for simple single-turn prompts

**Key difference from Claude:** Claude was specifically trained on XML-tagged prompts and has strong XML preference. Gemini is agnostic between XML and Markdown -- both work equally well as long as you're consistent.

---

## 3. System Instructions vs User Messages

Source: [System Instructions (cloud.google.com)](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions), [Live API Best Practices](https://ai.google.dev/gemini-api/docs/live-api/best-practices)

**What goes in system instructions:**
1. Persona/role definition (name, domain, characteristics)
2. Output format (JSON, Markdown, YAML, etc.)
3. Style and tone (verbosity, formality, reading level)
4. Task rules and constraints
5. Additional context (knowledge cutoff, domain knowledge)
6. Response language

**What goes in user messages:**
- The actual task/question
- Input data (images, text to analyze)
- Few-shot examples (can go in either, but system instructions keep them persistent across turns)

**Structure recommendation for system instructions:**
1. Persona definition first
2. Conversational rules / behavioral constraints second
3. Guardrails / safety rules third

**Important caveat:** System instructions don't fully prevent jailbreaks. Don't put secrets in them.

**Note for OpenRouter:** When calling via OpenRouter's chat completions API, system instructions map to the `system` role message. Place all persistent behavioral instructions there, and per-frame analysis requests in `user` messages.

---

## 4. Vision/Multimodal Prompt Tips

Source: [Image understanding (ai.google.dev)](https://ai.google.dev/gemini-api/docs/image-understanding), [File prompting strategies](https://ai.google.dev/gemini-api/docs/file-prompting-strategies), [Video understanding](https://ai.google.dev/gemini-api/docs/video-understanding)

**Image placement:**
- Single image: Place the image BEFORE the text prompt in the contents array (image first, then text).
- Multiple images: Supported, up to 3,600 image files per request.
- Mixed sources: Can combine File API uploads with inline base64 data.

**Image quality:**
- Ensure images are correctly rotated.
- Use clear, non-blurry images.
- For fine text or small details, use higher `media_resolution` setting (increases token cost).

**Multimodal prompting rules:**
- Treat text, images, audio, video as "equal-class inputs."
- Reference each modality explicitly in your instructions ("In this image...", "Based on the audio transcript...").
- Don't assume the model will synthesize modalities automatically -- tell it to.

**Vision-specific tips for our pipeline:**
- Base64 JPEG inline is fine for frames under 20MB total request size.
- For object detection/segmentation tasks, disable thinking mode (set thinking budget to 0).
- Place the frame image first in the content array, then the text analysis prompt after it.

**Few-shot for vision:**
- Gemini accepts multiple image+text pairs as few-shot examples.
- Format must be identical across examples.

---

## 5. JSON Output Best Practices

Source: [Structured outputs (ai.google.dev)](https://ai.google.dev/gemini-api/docs/structured-output), [Vertex AI Structured Output](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output)

**Two approaches, and they stack:**

### A. Constrained decoding (API-level, preferred):
Set `response_mime_type: "application/json"` and provide `response_json_schema` in the API request. This uses constrained decoding -- the model is guaranteed to produce syntactically valid JSON matching your schema.

Best practices:
- Use specific types (`integer`, `string`, `enum`) not generic ones.
- Use `enum` for fields with limited valid values (event types, sources).
- Write clear `description` fields in schema properties -- the model reads them.
- Don't duplicate the schema in the prompt text. Put it in `responseSchema` only.
- Keep schemas flat. Avoid deep nesting -- API may reject overly complex schemas.
- Shorten property names if hitting complexity limits.

### B. Prompt-based JSON (for APIs without schema support, like OpenRouter):
When you can't use `responseSchema` (e.g., through OpenRouter), put JSON format instructions in the prompt:

```
Respond with ONLY a raw JSON object. No markdown code fences. No explanation.
Format:
{"type": "step_completion", "step_id": 3, "confidence": 0.85}
```

Gemini 3 tips for prompt-based JSON:
- Name the output format in one line ("Respond as JSON").
- Remove redundant constraints from old Gemini 2.x prompts. Gemini 3 picks up structure from minimal cues.
- Add detail back only when output quality drops.
- Include "No markdown wrapping" or "Raw JSON only" to prevent triple-backtick wrapping.

**Semantic validation:** Constrained decoding guarantees valid JSON structure but NOT correct values. Always validate business logic in your application code.

---

## 6. Gemini vs GPT-4 Structured Prompt Differences

Source: [Context Studios comparison](https://www.contextstudios.ai/comparisons/gemini-vs-chatgpt-prompting), [Model-Specific Prompting (joanmedia.dev)](https://www.joanmedia.dev/ai-blog/model-specific-prompting-how-claude-gpt-and-gemini-differ), [philschmid.de](https://www.philschmid.de/gemini-3-prompt-practices)

| Aspect | Gemini 3 | GPT-4 / GPT-4o |
|--------|----------|-----------------|
| **Instruction style** | Direct, concise, no padding | Tolerates verbose, conversational instructions |
| **Inference from vague prompts** | Poor -- needs explicit instructions | Good at inferring intent from vague prompts |
| **Structured formatting** | XML or Markdown (pick one, be consistent) | JSON/Markdown preferred, XML works but less common |
| **System instructions** | Strong support, place constraints here | Strong support, similar usage |
| **Temperature** | Keep at 1.0 (below causes issues) | 0.0-0.7 common for deterministic tasks |
| **Chain-of-thought** | Use `thinking_level` parameter instead of prompt-based CoT | Explicit "think step by step" in prompt works well |
| **Output verbosity** | Default terse, must request detail | Default verbose, must constrain |
| **Long context** | Data first, question at end, use bridging phrases | More flexible about ordering |
| **JSON output** | Constrained decoding via `responseSchema` or prompt-based | `response_format: json_object` or JSON mode |
| **Multimodal** | Image before text in content array | Either order works, text-first common |
| **Few-shot** | Critical for good results, format consistency matters a lot | Helpful but model infers well without them |

**Key takeaway for our pipeline:** Gemini prompts should be shorter and more structured than GPT-4 prompts. Strip out explanatory text, use clear delimiters (XML tags or markdown headers), put the image first, and be explicit about the exact output format wanted.

---

## Action Items for Pipeline (`src/run.py`)

Based on these findings, the current VLM prompt (`prompts/vlm_prompt.txt`) should be revised:

1. **Image placement**: Ensure base64 frame is the first content part, text prompt second.
2. **Shorter prompt**: Gemini 3 prefers concise instructions. Cut explanatory padding.
3. **Consistent delimiters**: Pick XML tags or Markdown headers, use one throughout.
4. **Explicit JSON format**: Include "Respond with ONLY raw JSON" since we're going through OpenRouter (no `responseSchema` support).
5. **Few-shot examples**: Add 2-3 example frame analyses showing expected output format.
6. **System vs user split**: Move persistent instructions (role, constraints, output format) to system message. Keep per-frame data (procedure context, audio transcripts, current state) in user message.
7. **Temperature**: Keep at 1.0 (default) -- do not lower for Gemini 3.

---

## Sources

- [Prompt design strategies | ai.google.dev](https://ai.google.dev/gemini-api/docs/prompting-strategies)
- [Gemini 3 Developer Guide | ai.google.dev](https://ai.google.dev/gemini-api/docs/gemini-3)
- [Image understanding | ai.google.dev](https://ai.google.dev/gemini-api/docs/image-understanding)
- [Structured outputs | ai.google.dev](https://ai.google.dev/gemini-api/docs/structured-output)
- [File prompting strategies | ai.google.dev](https://ai.google.dev/gemini-api/docs/file-prompting-strategies)
- [Video understanding | ai.google.dev](https://ai.google.dev/gemini-api/docs/video-understanding)
- [System instructions | cloud.google.com](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/system-instructions)
- [Live API best practices | ai.google.dev](https://ai.google.dev/gemini-api/docs/live-api/best-practices)
- [Gemini 3 Prompting Best Practices | philschmid.de](https://www.philschmid.de/gemini-3-prompt-practices)
- [Gemini vs ChatGPT Prompting | contextstudios.ai](https://www.contextstudios.ai/comparisons/gemini-vs-chatgpt-prompting)
- [Model-Specific Prompting | joanmedia.dev](https://www.joanmedia.dev/ai-blog/model-specific-prompting-how-claude-gpt-and-gemini-differ)
- [Media resolution | ai.google.dev](https://ai.google.dev/gemini-api/docs/media-resolution)
