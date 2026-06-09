/**
 * Outline module – create/update outlines, generate search queries.
 * TypeScript port of outline_module.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import yaml from "js-yaml";
import type { LLMModel } from "./llmUtils.js";
import { callLlmModel } from "./llmUtils.js";
import { updateLlmUsage } from "./utils.js";
import type { EvidenceNode, KnowledgeGraph } from "./dataModel.js";
import { knowledgeGraphToJson } from "./dataModel.js";

const PROMPT_LIB_DIR = path.resolve(
  path.dirname(new URL(import.meta.url).pathname),
  "prompt_lib",
);

function loadPrompt(name: string): Record<string, string> {
  const raw = fs.readFileSync(path.join(PROMPT_LIB_DIR, name), "utf-8");
  return yaml.load(raw) as Record<string, string>;
}

// ─── createOutline ───────────────────────────────────────────────────────────

export async function createOutline(
  rootQuery: string,
  llmModel: LLMModel,
  language: string,
  reportId?: number,
  usageFile?: string,
): Promise<string> {
  const yamlData = loadPrompt("create_outline.yaml");
  const systemPrompt = yamlData.system;
  const userPrompt = `Root Query: ${rootQuery}\nLanguage: ${language}`;
  const messages = [
    { role: "system" as const, content: systemPrompt },
    { role: "user" as const, content: userPrompt },
  ];

  const start = Date.now();
  const response = await callLlmModel(llmModel, messages, {
    temperature: 0.7,
    num_retry: 3,
  });
  const elapsed = (Date.now() - start) / 1000;

  if (reportId !== undefined && usageFile) {
    updateLlmUsage(
      response,
      "create_outline",
      reportId,
      usageFile,
      response._call_elapsed_time || elapsed,
    );
  }

  return response.content;
}

// ─── updateOutline ───────────────────────────────────────────────────────────

export async function updateOutline(
  rootQuery: string,
  outline: string,
  evidences: EvidenceNode[],
  llmModel: LLMModel,
  language: string,
  reportId?: number,
  usageFile?: string,
): Promise<string> {
  const evidenceStr = evidences
    .map((en) => `id_${en.id}: ${en.content}`)
    .join("\n");

  const yamlData = loadPrompt("update_outline.yaml");
  const systemPrompt = yamlData.system;
  const userPrompt = `
    Root Query: ${rootQuery}
    
    Current Outline: ${outline}
    
    New Evidences: ${evidenceStr}
    
    Language: ${language}`;

  const messages = [
    { role: "system" as const, content: systemPrompt },
    { role: "user" as const, content: userPrompt },
  ];

  const start = Date.now();
  const response = await callLlmModel(llmModel, messages, {
    temperature: 0.7,
  });
  const elapsed = (Date.now() - start) / 1000;

  if (reportId !== undefined && usageFile) {
    updateLlmUsage(
      response,
      "update_outline",
      reportId,
      usageFile,
      response._call_elapsed_time || elapsed,
    );
  }

  return response.content;
}

// ─── updateOutlineByKg ───────────────────────────────────────────────────────

export async function updateOutlineByKg(
  rootQuery: string,
  outline: string,
  evidences: EvidenceNode[],
  knowledgeGraph: KnowledgeGraph,
  llmModel: LLMModel,
  language: string,
  reportId?: number,
  usageFile?: string,
): Promise<string> {
  const evidenceStr = evidences
    .map((en) => `id_${en.id}: ${en.content}`)
    .join("\n");

  const yamlData = loadPrompt("update_outline_by_kg.yaml");
  const systemPrompt = yamlData.system;
  const userPrompt = `
    Root Query: ${rootQuery}
    
    Current Outline: ${outline}
    
    New Evidences: ${evidenceStr}
    
    Current Knowledge Graph: ${JSON.stringify(knowledgeGraphToJson(knowledgeGraph))}
    
    Language: ${language}`;

  const messages = [
    { role: "system" as const, content: systemPrompt },
    { role: "user" as const, content: userPrompt },
  ];

  const start = Date.now();
  const response = await callLlmModel(llmModel, messages, {
    temperature: 0.7,
  });
  const elapsed = (Date.now() - start) / 1000;

  if (reportId !== undefined && usageFile) {
    updateLlmUsage(
      response,
      "update_outline_by_kg",
      reportId,
      usageFile,
      response._call_elapsed_time || elapsed,
    );
  }

  return response.content;
}

// ─── generateSearchQueries ───────────────────────────────────────────────────

export async function generateSearchQueries(
  outline: string,
  historySearchQueries: string[],
  llmModel: LLMModel,
  language: string,
  options?: {
    reportId?: number;
    usageFile?: string;
    isUseHistorySearchQueries?: boolean;
    pendingSearchQueries?: string[];
    queryNum?: number;
  },
): Promise<string[]> {
  const queryNum = options?.queryNum ?? 10;
  const yamlData = loadPrompt("generate_search_query.yaml");
  const systemPrompt = yamlData.system.replace(
    "${QUERY_NUM}",
    String(queryNum),
  );

  const userLines: string[] = [`Outline: ${outline}`];
  if (options?.isUseHistorySearchQueries && historySearchQueries.length > 0) {
    userLines.push(
      `Historical Search Queries (executed):${JSON.stringify(historySearchQueries)}`,
    );
  }
  if (options?.pendingSearchQueries?.length) {
    userLines.push(
      `Pending Search Queries (planned, NOT executed yet):${JSON.stringify(options.pendingSearchQueries)}`,
    );
  }
  userLines.push(`Language: ${language}`);
  const userPrompt = userLines.join("\n\n");

  const messages = [
    { role: "system" as const, content: systemPrompt },
    { role: "user" as const, content: userPrompt },
  ];

  const start = Date.now();
  const response = await callLlmModel(llmModel, messages, {
    temperature: 0.7,
  });
  const elapsed = (Date.now() - start) / 1000;

  if (options?.reportId !== undefined && options?.usageFile) {
    updateLlmUsage(
      response,
      "generate_search_queries",
      options.reportId,
      options.usageFile,
      response._call_elapsed_time || elapsed,
    );
  }

  return response.content
    .trim()
    .split("\n")
    .filter((q) => q.trim() !== "");
}
