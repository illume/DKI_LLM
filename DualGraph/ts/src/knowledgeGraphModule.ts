/**
 * Knowledge graph module – KG extraction, merging, and search-chain generation.
 * TypeScript port of knowledge_graph_module.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";
import type { Chain, EvidenceNode, KnowledgeGraph } from "./dataModel.ts";
import {
  addLlmGeneratedKnowledge,
  applyMergeNodeResults,
  getKnowledgeNodeById,
  knowledgeGraphToText,
} from "./dataModel.ts";
import type { LLMModel } from "./llmUtils.ts";
import { callLlmModel } from "./llmUtils.ts";
import { safeJsonLoads, updateLlmUsage } from "./utils.ts";

const PROMPT_LIB_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "prompt_lib",
);

function loadPrompt(name: string): Record<string, string> {
  const raw = fs.readFileSync(path.join(PROMPT_LIB_DIR, name), "utf-8");
  return yaml.load(raw) as Record<string, string>;
}

interface ExtractKnowledgeResult {
  new_nodes?: Array<{
    id: string;
    node_name: string;
    is_core_entity: boolean;
  }>;
  new_edges?: Array<{
    id: string;
    source_id: string;
    target_id: string;
    relation_name: string;
  }>;
  evidences_map?: Record<string, string[]>;
}

interface MergeNodeResult {
  clusters?: Array<{
    cluster_id: string;
    representative_concept: string;
    source_node_ids: string[];
    similarity_justification: string;
  }>;
}

interface SelectExploreChainsResult {
  chains?: unknown[];
  "search queries"?: unknown[];
}

export interface GenerateSearchChainsResult {
  selectedChains: Chain[];
  searchQueries: string[];
  allChains: Chain[];
  updatedVisitedEdges: Set<string>;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
}

function parseExtractKnowledgeResult(value: unknown): ExtractKnowledgeResult {
  const record = asRecord(value);
  if (!record) {
    throw new Error("LLM did not return a JSON object for knowledge extraction");
  }
  return record as unknown as ExtractKnowledgeResult;
}

function parseMergeNodeResult(value: unknown): MergeNodeResult {
  const record = asRecord(value);
  if (!record) {
    throw new Error("LLM did not return a JSON object for merge results");
  }
  return record as unknown as MergeNodeResult;
}

function parseSelectExploreChainsResult(value: unknown): SelectExploreChainsResult {
  const record = asRecord(value);
  if (!record) {
    throw new Error("LLM did not return a JSON object for explore chains");
  }
  return record as unknown as SelectExploreChainsResult;
}

function formatEvidenceNodes(evidenceNodes: EvidenceNode[]): string {
  return evidenceNodes
    .map((node, index) => {
      const evidenceId = node.id ?? index + 1;
      const title = node.source_title?.trim() || "Untitled";
      const content = node.content?.trim() || "";
      return `EN${evidenceId}: Title: ${title}\nContent: ${content}`;
    })
    .join("\n\n");
}

function maybeTrackUsage(
  response: Awaited<ReturnType<typeof callLlmModel>>,
  moduleName: string,
  reportId?: number,
  usageFile?: string,
): void {
  if (reportId === undefined || !usageFile) return;
  updateLlmUsage(
    response,
    moduleName,
    reportId,
    usageFile,
    response._call_elapsed_time,
  );
}

function buildKnowledgeExtractionPrompt(
  rootQuery: string,
  searchQuery: string,
  knowledgeGraph: KnowledgeGraph,
  evidenceNodes: EvidenceNode[],
): string {
  const parts = [
    `Root Query: ${rootQuery}`,
    `Search Query: ${searchQuery}`,
    `Evidence Statements:\n${formatEvidenceNodes(evidenceNodes)}`,
  ];

  if (knowledgeGraph.knowledge_nodes.length > 0 || knowledgeGraph.knowledge_edges.length > 0) {
    parts.push(`Current Knowledge Graph:\n${knowledgeGraphToText(knowledgeGraph)}`);
  }

  return parts.join("\n\n");
}

async function runKnowledgeExtraction(
  promptName: string,
  usageModuleName: string,
  rootQuery: string,
  knowledgeGraph: KnowledgeGraph,
  evidenceNodes: EvidenceNode[],
  searchQuery: string,
  llmModel: LLMModel,
  reportId?: number,
  usageFile?: string,
): Promise<ExtractKnowledgeResult> {
  const yamlData = loadPrompt(promptName);
  const response = await callLlmModel(
    llmModel,
    [
      { role: "system", content: yamlData.system ?? "" },
      {
        role: "user",
        content: buildKnowledgeExtractionPrompt(
          rootQuery,
          searchQuery,
          knowledgeGraph,
          evidenceNodes,
        ),
      },
    ],
    { temperature: 0.3, num_retry: 3 },
  );

  maybeTrackUsage(response, usageModuleName, reportId, usageFile);
  return parseExtractKnowledgeResult(safeJsonLoads(response.content));
}

async function runMergeStep(
  rootQuery: string,
  knowledgeGraph: KnowledgeGraph,
  llmModel: LLMModel,
  reportId?: number,
  usageFile?: string,
): Promise<void> {
  if (knowledgeGraph.knowledge_nodes.length < 2) return;

  const yamlData = loadPrompt("merge_knowledge_node.yaml");
  const response = await callLlmModel(
    llmModel,
    [
      { role: "system", content: yamlData.system ?? "" },
      {
        role: "user",
        content: `Root Query: ${rootQuery}\n\nCurrent Knowledge Graph:\n${knowledgeGraphToText(knowledgeGraph)}`,
      },
    ],
    { temperature: 0.3, num_retry: 3 },
  );

  maybeTrackUsage(response, "merge_knowledge_node", reportId, usageFile);
  applyMergeNodeResults(
    knowledgeGraph,
    parseMergeNodeResult(safeJsonLoads(response.content)),
  );
}

export async function createKnowledgeGraph(
  rootQuery: string,
  knowledgeGraph: KnowledgeGraph,
  evidenceNodesBatch: EvidenceNode[][],
  searchQueries: string[],
  llmModel: LLMModel,
  reportId?: number,
  usageFile?: string,
): Promise<KnowledgeGraph> {
  if (evidenceNodesBatch.length !== searchQueries.length) {
    throw new Error("Mismatch in evidenceNodesBatch and searchQueries length");
  }

  for (let i = 0; i < evidenceNodesBatch.length; i += 1) {
    const evidenceNodes = evidenceNodesBatch[i] ?? [];
    const searchQuery = searchQueries[i] ?? "";
    if (evidenceNodes.length === 0) continue;

    const extracted = await runKnowledgeExtraction(
      "extract_knowledge_node.yaml",
      "extract_knowledge_node",
      rootQuery,
      knowledgeGraph,
      evidenceNodes,
      searchQuery,
      llmModel,
      reportId,
      usageFile,
    );
    addLlmGeneratedKnowledge(knowledgeGraph, extracted, evidenceNodes);
    await runMergeStep(rootQuery, knowledgeGraph, llmModel, reportId, usageFile);
  }

  return knowledgeGraph;
}

export async function updateKnowledgeGraph(
  rootQuery: string,
  knowledgeGraph: KnowledgeGraph,
  evidenceNodesBatch: EvidenceNode[][],
  searchQueries: string[],
  llmModel: LLMModel,
  reportId?: number,
  usageFile?: string,
): Promise<KnowledgeGraph> {
  if (evidenceNodesBatch.length !== searchQueries.length) {
    throw new Error("Mismatch in evidenceNodesBatch and searchQueries length");
  }

  for (let i = 0; i < evidenceNodesBatch.length; i += 1) {
    const evidenceNodes = evidenceNodesBatch[i] ?? [];
    const searchQuery = searchQueries[i] ?? "";
    if (evidenceNodes.length === 0) continue;

    const extracted = await runKnowledgeExtraction(
      "update_knowledge_node.yaml",
      "update_knowledge_node",
      rootQuery,
      knowledgeGraph,
      evidenceNodes,
      searchQuery,
      llmModel,
      reportId,
      usageFile,
    );
    addLlmGeneratedKnowledge(knowledgeGraph, extracted, evidenceNodes);
    await runMergeStep(rootQuery, knowledgeGraph, llmModel, reportId, usageFile);
  }

  return knowledgeGraph;
}

function visitedEdgeKey(sourceId: number, targetId: number, type: string): string {
  return `${sourceId}:${targetId}:${type}`;
}

function relationExists(
  knowledgeGraph: KnowledgeGraph,
  sourceId: number,
  targetId: number,
): boolean {
  return knowledgeGraph.knowledge_edges.some(
    (edge) =>
      (edge.source_id === sourceId && edge.target_id === targetId) ||
      (edge.source_id === targetId && edge.target_id === sourceId),
  );
}

function buildChainContent(
  knowledgeGraph: KnowledgeGraph,
  sourceId: number,
  targetId: number,
  relationName: string,
): string {
  const sourceName = getKnowledgeNodeById(knowledgeGraph, sourceId)?.knowledge ?? "?";
  const targetName = getKnowledgeNodeById(knowledgeGraph, targetId)?.knowledge ?? "?";
  return `(${sourceName}) -> [${relationName}] -> (${targetName})`;
}

function generateEnrichChains(
  knowledgeGraph: KnowledgeGraph,
  visitedEdges: Set<string>,
  startId: number,
): Chain[] {
  const chains: Chain[] = [];
  let nextId = startId;

  for (const edge of knowledgeGraph.knowledge_edges) {
    if (edge.evidence_nodes.length >= 2) continue;
    const key = visitedEdgeKey(edge.source_id, edge.target_id, "enrich");
    if (visitedEdges.has(key)) continue;

    chains.push({
      id: nextId,
      type: "enrich",
      nodes: [edge.source_id, edge.target_id],
      content: buildChainContent(
        knowledgeGraph,
        edge.source_id,
        edge.target_id,
        edge.relation_name,
      ),
      reason: `Low evidence count (${edge.evidence_nodes.length})`,
      is_visited: false,
    });
    nextId += 1;
  }

  return chains;
}

function generateEntityConceptExploreChains(
  knowledgeGraph: KnowledgeGraph,
  visitedEdges: Set<string>,
  startId: number,
): Chain[] {
  const chains: Chain[] = [];
  let nextId = startId;
  const coreEntities = knowledgeGraph.knowledge_nodes.filter((node) => node.is_core_entity);
  const concepts = knowledgeGraph.knowledge_nodes.filter((node) => !node.is_core_entity);

  for (const entity of coreEntities) {
    for (const concept of concepts) {
      if (relationExists(knowledgeGraph, entity.id, concept.id)) continue;
      const key = visitedEdgeKey(entity.id, concept.id, "explore_entity_concept");
      if (visitedEdges.has(key)) continue;

      const conceptDegree = knowledgeGraph.knowledge_edges.filter(
        (edge) => edge.source_id === concept.id || edge.target_id === concept.id,
      ).length;
      if (conceptDegree === 0) continue;

      chains.push({
        id: nextId,
        type: "explore_entity_concept",
        nodes: [entity.id, concept.id],
        content: buildChainContent(knowledgeGraph, entity.id, concept.id, "?"),
        reason: `Missing direct link between core entity and concept (concept degree: ${conceptDegree})`,
        is_visited: false,
      });
      nextId += 1;
    }
  }

  return chains;
}

function normalizeSelectedChainIds(value: unknown[], validIds: Set<number>): number[] {
  const result: number[] = [];
  const seen = new Set<number>();

  for (const item of value) {
    let numericId: number | null = null;
    if (typeof item === "number" && Number.isFinite(item)) {
      numericId = item;
    } else if (typeof item === "string") {
      const cleaned = item.trim().replace(/^[cC]/, "");
      const parsed = Number.parseInt(cleaned, 10);
      if (Number.isFinite(parsed)) numericId = parsed;
    }

    if (numericId !== null && validIds.has(numericId) && !seen.has(numericId)) {
      seen.add(numericId);
      result.push(numericId);
    }
  }

  return result;
}

function renderExplorePrompt(
  rootQuery: string,
  outline: string,
  knowledgeGraph: KnowledgeGraph,
  chains: Chain[],
  language: string,
  isUseHistorySearchQueries: boolean,
  historySearchQueries: string[],
): string {
  const lines = [
    `Root Query: ${rootQuery}`,
    `Current Outline:\n${outline}`,
    `Knowledge Graph:\n${knowledgeGraphToText(knowledgeGraph)}`,
  ];

  if (isUseHistorySearchQueries && historySearchQueries.length > 0) {
    lines.push(`History Search Queries:\n${JSON.stringify(historySearchQueries)}`);
  }

  const chainLines = chains.map(
    (chain) => `${chain.id}. ${chain.content ?? ""} -- Reason: ${chain.reason}`,
  );
  lines.push(`Explore Chains:\n${chainLines.join("\n")}`);
  lines.push(`Generate Search Queries in ${language}.`);
  return lines.join("\n\n");
}

export async function generateSearchChainsAndSearchQueries(
  rootQuery: string,
  outline: string,
  knowledgeGraph: KnowledgeGraph,
  llmModel: LLMModel,
  visitedEdges: Set<string> = new Set(),
  language: string = "en",
  reportId?: number,
  usageFile?: string,
  isUseHistorySearchQueries: boolean = false,
  historySearchQueries: string[] = [],
  kgQueryNum: number = 10,
): Promise<GenerateSearchChainsResult> {
  const enrichChains = generateEnrichChains(knowledgeGraph, visitedEdges, 1);
  const exploreChains = generateEntityConceptExploreChains(
    knowledgeGraph,
    visitedEdges,
    enrichChains.length + 1,
  );
  const allChains = [...enrichChains, ...exploreChains];

  if (allChains.length === 0) {
    return {
      selectedChains: [],
      searchQueries: [],
      allChains: [],
      updatedVisitedEdges: new Set(visitedEdges),
    };
  }

  const yamlData = loadPrompt("select_explore_chains.yaml");
  const systemPrompt = (yamlData.system ?? "").replaceAll(
    "${CHAIN_NUM}",
    String(kgQueryNum),
  );
  const response = await callLlmModel(
    llmModel,
    [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: renderExplorePrompt(
          rootQuery,
          outline,
          knowledgeGraph,
          allChains,
          language,
          isUseHistorySearchQueries,
          historySearchQueries,
        ),
      },
    ],
    { temperature: 0.7, num_retry: 3 },
  );

  maybeTrackUsage(response, "generate_explore_queries", reportId, usageFile);

  const parsed = parseSelectExploreChainsResult(safeJsonLoads(response.content));
  const rawChainIds = Array.isArray(parsed.chains) ? parsed.chains : [];
  const rawSearchQueries = Array.isArray(parsed["search queries"])
    ? parsed["search queries"]
    : [];

  const validIds = new Set(allChains.map((chain) => chain.id));
  const selectedIds = normalizeSelectedChainIds(rawChainIds, validIds);
  const selectedChains =
    selectedIds.length > 0
      ? allChains.filter((chain) => selectedIds.includes(chain.id))
      : allChains.slice(0, Math.min(kgQueryNum, allChains.length));

  const fallbackQueries = selectedChains.map((chain) => chain.content ?? chain.reason);
  const searchQueries = rawSearchQueries
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .slice(0, kgQueryNum);

  const finalQueries =
    searchQueries.length > 0
      ? searchQueries.slice(0, Math.max(selectedChains.length, searchQueries.length))
      : fallbackQueries.slice(0, kgQueryNum);

  const updatedVisitedEdges = new Set(visitedEdges);
  for (const chain of selectedChains) {
    chain.is_visited = true;
    if (chain.nodes.length >= 2) {
      updatedVisitedEdges.add(
        visitedEdgeKey(chain.nodes[0], chain.nodes[chain.nodes.length - 1], chain.type),
      );
    }
  }

  return {
    selectedChains,
    searchQueries: finalQueries,
    allChains,
    updatedVisitedEdges,
  };
}
