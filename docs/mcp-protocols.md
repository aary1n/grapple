# MCP Tooling Protocols

## Overview
Grapple development requires precise tool usage. **Never guess** library APIs, database schemas, or file structures. Use MCP tools to fetch authoritative information.

---

## 1. Context7 (Documentation Fetching)

### When to Use
**TRIGGER on these patterns:**
- Mentioning specific library versions (e.g., ".NET 9", "MediaPipe 0.10.x", "FlashCap")
- Working with C# language features (`Span<byte>`, `unsafe`, `Interlocked`, `readonly struct`)
- Python dependency operations (`numpy`, `mediapipe`)
- ML/AI library usage (`torch`, `transformers`, `onnxruntime`, `peft`, `autoawq`, `timm`)
- Questions about API signatures, best practices, or version-specific behavior

**ACTION:**
```
Explicitly call: use context7 to fetch docs for [Library + Version]
```

**Examples:**
- "I need to optimize Span<byte> usage" → `context7: .NET 9 Span documentation`
- "How do I configure FlashCap?" → `context7: FlashCap API reference`
- "MediaPipe hand landmark structure" → `context7: MediaPipe 0.10 hand tracking`
- "LoRA adapter fine-tuning" → `context7: PEFT LoRA configuration`
- "ONNX Runtime DirectML provider" → `context7: ONNX Runtime DirectML execution provider`
- "AWQ quantization" → `context7: AutoAWQ quantization API`
- "PyTorch model export" → `context7: PyTorch ONNX export`

**CONSTRAINT:**
- Do NOT infer API signatures from memory or old examples
- If uncertain about a method signature, property, or pattern → **fetch docs first**

**FALLBACK:**
- If Context7 fails or returns insufficient info → Ask user for official docs URL
- Then use `read` tool to fetch the specific page

---

## 2. Filesystem (Code Navigation & Analysis)

### When to Use
**TRIGGER on these patterns:**
- "Where is [component] implemented?"
- "Show me the current [service/class/module]"
- Before suggesting refactors or architectural changes
- When debugging requires seeing actual file structure

**ACTION:**
```
1. Use filesystem to list directories and locate relevant files
2. Read specific files to understand current implementation
3. Only then propose changes based on ACTUAL code, not assumptions
```

**CONSTRAINT:**
- Do NOT suggest code changes without first reading existing implementation
- Do NOT assume project structure matches typical patterns

**PROHIBITED:**
- "I assume you have a Services folder..." ❌
- "Typically this would be in..." ❌
- CORRECT: "Let me check your current structure..." ✅

---

## 3. PostgreSQL (Database Schema Authority)

### When to Use
**TRIGGER on these patterns:**
- Designing database migrations
- Writing complex queries (JOINs, CTEs, aggregations)
- Performance optimization requiring index analysis
- Schema questions ("What columns does X have?")

**ACTION:**
```
1. Connect to postgresql via MCP
2. Introspect schema: \d table_name, \di (indexes), \df (functions)
3. Base ALL query/migration work on live schema
```

**CONSTRAINT:**
- Do NOT infer schema from old migration files or code comments
- Do NOT assume column names, types, or constraints

---

## Tool Priority Matrix

| Scenario | Primary Tool | Secondary Tool | Fallback |
|----------|--------------|----------------|----------|
| API usage question | Context7 | User-provided URL + read | Documentation comment |
| Code structure question | Filesystem | - | Ask user |
| Database query | PostgreSQL introspection | - | Ask user for schema |
| Library version conflict | Context7 (specific version) | Release notes URL | Ask user |

---

## Anti-Patterns

### ❌ DON'T:
```
"In .NET, you typically use Task.Run for async work..."
(Without checking .NET 9 specific patterns)
```

### ✅ DO:
```
"Let me fetch .NET 9 async best practices first."
[calls context7: .NET 9 Task and async/await patterns]
```

---

## Zero-Tolerance Rules

1. **Never guess library APIs** - If you don't have Context7 docs, explicitly say "I need to fetch documentation for [X] before proceeding"
2. **Never assume schema** - If you need DB structure, say "I need to inspect the database schema first"
3. **Never fake file paths** - If you need to know where something is, say "Let me navigate your filesystem to locate [X]"
4. **When in doubt, fetch** - It's better to make 3 tool calls and be accurate than make 0 calls and be wrong

---

## Integration with Grapple Workflow

Since Grapple is performance-critical (zero-GC, ultra-low-latency), tool usage is especially important:

- **Before suggesting `Span<T>` usage** → Context7: .NET 9 Span best practices
- **Before modifying IPC layer** → Filesystem: read current shared memory implementation
- **Before query optimization** → PostgreSQL: check actual query plans with EXPLAIN ANALYZE
- **Before dependency updates** → Context7: check version compatibility for MediaPipe, FlashCap

---

## Quick Reference

```
Library API unclear?        → context7 [library] [version]
Don't know file location?   → filesystem list/read
Need schema info?           → postgresql \d [table]
Documentation link broken?  → read [url]
```
