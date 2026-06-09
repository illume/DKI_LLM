/**
 * Utility helpers – usage tracking, JSON repair, file I/O.
 * TypeScript port of utils_module.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { jsonrepair } from "jsonrepair";
import type { ModelResponse } from "./llmUtils.ts";

// ─── Usage tracking ──────────────────────────────────────────────────────────

/**
 * Thread-safe in Python; in Node single-threaded so a simple read-modify-write
 * is sufficient. If future versions use workers, wrap with a mutex.
 */

function readUsageFile(usageFile: string): Record<string, unknown> {
  if (!fs.existsSync(usageFile)) return {};
  try {
    return JSON.parse(fs.readFileSync(usageFile, "utf-8"));
  } catch {
    return {};
  }
}

function writeUsageFile(
  usageFile: string,
  data: Record<string, unknown>,
): void {
  fs.mkdirSync(path.dirname(usageFile), { recursive: true });
  const tmp = usageFile + ".tmp";
  fs.writeFileSync(tmp, JSON.stringify(data, null, 4), "utf-8");
  fs.renameSync(tmp, usageFile);
}

export function clearUsageDataIfExists(
  reportId: number,
  usageFile: string,
): void {
  const config = readUsageFile(usageFile);
  const rid = String(reportId);
  let needSave = false;

  for (const key of [
    "llm_usage",
    "filter_stats",
    "module_elapsed_time",
    "action_elapsed_time",
    "readpage_stats",
  ]) {
    const section = config[key] as Record<string, unknown> | undefined;
    if (section && rid in section) {
      delete section[rid];
      needSave = true;
    }
  }

  if (needSave) writeUsageFile(usageFile, config);
}

export function updateLlmUsage(
  response: ModelResponse,
  moduleName: string,
  reportId: number,
  usageFile: string,
  elapsedTime: number,
): void {
  const config = readUsageFile(usageFile);
  const rid = String(reportId);

  if (!config.llm_usage) config.llm_usage = {};
  const usage = config.llm_usage as Record<string, Record<string, unknown>>;
  if (!usage[rid]) usage[rid] = {};
  const report = usage[rid] as Record<
    string,
    { calls: number; prompt_tokens: number; completion_tokens: number; total_tokens: number; elapsed_time: number }
  >;
  if (!report[moduleName]) {
    report[moduleName] = {
      calls: 0,
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
      elapsed_time: 0,
    };
  }
  const m = report[moduleName];
  m.calls += 1;
  m.prompt_tokens += response.usage.prompt_tokens;
  m.completion_tokens += response.usage.completion_tokens;
  m.total_tokens += response.usage.total_tokens;
  m.elapsed_time += elapsedTime;

  writeUsageFile(usageFile, config);
}

export function updateElapsedTime(
  reportId: number,
  usageFile: string,
  elapsedTime: number,
): void {
  const config = readUsageFile(usageFile);
  const rid = String(reportId);
  if (!config.module_elapsed_time) config.module_elapsed_time = {};
  (config.module_elapsed_time as Record<string, number>)[rid] = elapsedTime;
  writeUsageFile(usageFile, config);
}

export function updateActionElapsedTime(
  actionName: string,
  reportId: number,
  usageFile: string,
  elapsedTime: number,
): void {
  const config = readUsageFile(usageFile);
  const rid = String(reportId);
  if (!config.action_elapsed_time) config.action_elapsed_time = {};
  const actions = config.action_elapsed_time as Record<string, Record<string, number>>;
  if (!actions[rid]) actions[rid] = {};
  actions[rid][actionName] =
    (actions[rid][actionName] ?? 0) + elapsedTime;
  writeUsageFile(usageFile, config);
}

export function updateFilterStats(
  reportId: number,
  usageFile: string,
  filterType: string,
  beforeCount: number,
  afterCount: number,
): void {
  const config = readUsageFile(usageFile);
  const rid = String(reportId);
  if (!config.filter_stats) config.filter_stats = {};
  const stats = config.filter_stats as Record<string, Record<string, unknown>>;
  if (!stats[rid]) stats[rid] = {};
  const report = stats[rid] as Record<string, { total_before: number; total_after: number; count: number }>;
  if (!report[filterType]) {
    report[filterType] = { total_before: 0, total_after: 0, count: 0 };
  }
  report[filterType].total_before += beforeCount;
  report[filterType].total_after += afterCount;
  report[filterType].count += 1;
  writeUsageFile(usageFile, config);
}

export function updateReadpageStats(
  reportId: number,
  usageFile: string,
  method: string,
  success: boolean,
): void {
  const config = readUsageFile(usageFile);
  const rid = String(reportId);
  if (!config.readpage_stats) config.readpage_stats = {};
  const stats = config.readpage_stats as Record<string, Record<string, { success: number; fail: number }>>;
  if (!stats[rid]) stats[rid] = {};
  if (!stats[rid][method]) stats[rid][method] = { success: 0, fail: 0 };
  if (success) stats[rid][method].success += 1;
  else stats[rid][method].fail += 1;
  writeUsageFile(usageFile, config);
}

// ─── JSON repair ─────────────────────────────────────────────────────────────

export function safeJsonLoads(text: string): unknown {
  // Step 1: Remove markdown code fences (```json ... ``` or ``` ... ```)
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  let cleaned = codeBlockMatch ? codeBlockMatch[1] : text.trim();

  // Step 2: Extract JSON substring between first {/[ and last }/]
  const startCurly = cleaned.indexOf("{");
  const startBracket = cleaned.indexOf("[");
  let startPos = -1;
  if (startCurly !== -1 && (startBracket === -1 || startCurly < startBracket)) {
    startPos = startCurly;
  } else if (startBracket !== -1) {
    startPos = startBracket;
  }

  const endCurly = cleaned.lastIndexOf("}");
  const endBracket = cleaned.lastIndexOf("]");
  const endPos = Math.max(endCurly, endBracket);

  if (startPos !== -1 && endPos !== -1 && endPos > startPos) {
    cleaned = cleaned.slice(startPos, endPos + 1);
  }

  try {
    return JSON.parse(cleaned);
  } catch {
    try {
      const repaired = jsonrepair(cleaned);
      return JSON.parse(repaired);
    } catch {
      return null;
    }
  }
}

// ─── Dedup helper ────────────────────────────────────────────────────────────

export function dedupPreserveOrder(items: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of items) {
    if (item && !seen.has(item)) {
      seen.add(item);
      result.push(item);
    }
  }
  return result;
}

// ─── Atomic write ────────────────────────────────────────────────────────────

export function atomicWriteText(
  filePath: string,
  content: string,
): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmp = filePath + ".tmp";
  fs.writeFileSync(tmp, content, "utf-8");
  fs.renameSync(tmp, filePath);
}
