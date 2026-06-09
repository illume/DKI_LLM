/**
 * LangGraph state graph – orchestrates the DualGraph research pipeline.
 *
 * 3-node graph:
 *   initialize → iterate ──(conditional)──▶ writeReport
 *                    ↻ (loop back if not terminal and iterations remain)
 */

import { Annotation, StateGraph, END } from "@langchain/langgraph";
import type { LLMModel } from "./llmUtils.ts";
import type { KnowledgeGraph, EvidenceNode, Chain } from "./dataModel.ts";
import { makeKnowledgeGraph, addEvidenceNode } from "./dataModel.ts";
import * as outlineModule from "./outlineModule.ts";
import * as knowledgeGraphModule from "./knowledgeGraphModule.ts";
import * as searchModule from "./searchModule.ts";
import * as writeModule from "./writeModule.ts";
import { judgeTerminalByOutline } from "./terminalModule.ts";
import {
  clearUsageDataIfExists,
  updateActionElapsedTime,
  updateElapsedTime,
  dedupPreserveOrder,
  atomicWriteText,
} from "./utils.ts";
import * as fs from "node:fs";
import * as path from "node:path";

// ─── Run configuration ───────────────────────────────────────────────────────

export interface RunConfig {
  kgQueryNum: number;
  ogQueryNum: number;
  searchProvider: string;
}

// ─── LangGraph State Annotation ──────────────────────────────────────────────

const ResearchState = Annotation.Root({
  // Input parameters (set once)
  reportId: Annotation<number>(),
  rootQuery: Annotation<string>(),
  reportDir: Annotation<string>(),
  productDir: Annotation<string>(),
  language: Annotation<string>(),
  usageFile: Annotation<string>(),
  maxIter: Annotation<number>(),
  disableEarlyStopping: Annotation<boolean>(),
  blockedUrls: Annotation<string[]>(),
  cfg: Annotation<RunConfig>(),

  // LLM model name (we reconstruct model per invocation for serializability)
  llmModelName: Annotation<string>(),

  // Mutable research state
  outline: Annotation<string>(),
  knowledgeGraphJson: Annotation<string>(), // JSON-serialized KnowledgeGraph
  historySearchQueries: Annotation<string[]>(),
  visitedUrls: Annotation<string[]>(),
  searchQueries: Annotation<string[]>(), // queries for next iteration
  visitedEdges: Annotation<string[]>(), // edge keys "src-tgt"
  iterationIndex: Annotation<number>(),
  isTerminal: Annotation<boolean>(),
  startTime: Annotation<number>(),

  // Output
  report: Annotation<string>(),
  success: Annotation<boolean>(),
  error: Annotation<string>(),
});

type ResearchStateType = typeof ResearchState.State;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function serializeKg(kg: KnowledgeGraph): string {
  return JSON.stringify(kg);
}

function deserializeKg(json: string): KnowledgeGraph {
  return JSON.parse(json) as KnowledgeGraph;
}

function visitedEdgesSetFromArray(arr: string[]): Set<string> {
  return new Set(arr);
}

function visitedEdgesArrayFromSet(s: Set<string>): string[] {
  return [...s];
}

// ─── Node: initialize ────────────────────────────────────────────────────────

async function initializeNode(
  state: ResearchStateType,
  llmModel: LLMModel,
): Promise<Partial<ResearchStateType>> {
  const {
    reportId,
    rootQuery,
    productDir,
    language,
    usageFile,
    cfg,
  } = state;

  clearUsageDataIfExists(reportId, usageFile);

  const visitedUrls = new Set<string>(state.blockedUrls || []);

  console.log(`\n${"=".repeat(80)}`);
  console.log(`[Start] Report ID: ${reportId}`);
  console.log(`${"=".repeat(80)}\n`);

  // Step 1: Create outline
  const createOutlineStart = Date.now();
  const outline = await outlineModule.createOutline(
    rootQuery,
    llmModel,
    language,
    reportId,
    usageFile,
  );
  updateActionElapsedTime(
    "create_outline",
    reportId,
    usageFile,
    (Date.now() - createOutlineStart) / 1000,
  );
  fs.mkdirSync(productDir, { recursive: true });
  fs.writeFileSync(
    path.join(productDir, `outline_case${reportId}_iter0.txt`),
    outline,
    "utf-8",
  );

  // Step 2: Generate search queries from outline
  const genSqStart = Date.now();
  let outlineSearchQueries = await outlineModule.generateSearchQueries(
    outline,
    [],
    llmModel,
    language,
    {
      reportId,
      usageFile,
      isUseHistorySearchQueries: false,
      queryNum: cfg.ogQueryNum,
    },
  );
  updateActionElapsedTime(
    "generate_search_queries",
    reportId,
    usageFile,
    (Date.now() - genSqStart) / 1000,
  );
  outlineSearchQueries = dedupPreserveOrder(outlineSearchQueries);

  // Step 3: Execute search
  let kg = makeKnowledgeGraph();
  const evidenceNodesBatch: EvidenceNode[][] = [];
  const searchIter0Start = Date.now();
  for (const searchQuery of outlineSearchQueries) {
    const newEvidences = await searchModule.searchWithFilteringVisitedUrls(
      searchQuery,
      rootQuery,
      llmModel,
      language,
      visitedUrls,
      reportId,
      usageFile,
      cfg.searchProvider as "bing" | "serper",
    );
    const batch: EvidenceNode[] = [];
    for (const ev of newEvidences) {
      const node = addEvidenceNode(kg, ev.source_title, ev.source_url, ev.content);
      visitedUrls.add(ev.source_url);
      batch.push(node);
    }
    evidenceNodesBatch.push(batch);
  }
  updateActionElapsedTime(
    "search_iter0",
    reportId,
    usageFile,
    (Date.now() - searchIter0Start) / 1000,
  );

  const historySearchQueries = [...outlineSearchQueries];

  // Step 4: Build knowledge graph
  const createKgStart = Date.now();
  kg = await knowledgeGraphModule.createKnowledgeGraph(
    rootQuery,
    kg,
    evidenceNodesBatch,
    outlineSearchQueries,
    llmModel,
    reportId,
    usageFile,
  );
  updateActionElapsedTime(
    "create_knowledge_graph",
    reportId,
    usageFile,
    (Date.now() - createKgStart) / 1000,
  );

  fs.writeFileSync(
    path.join(productDir, `knowledge_graph_case${reportId}_iter0.txt`),
    JSON.stringify(kg),
    "utf-8",
  );

  // Step 5: Generate explore queries from KG
  const genExploreStart = Date.now();
  const {
    selectedChains: thisTimeSearchChains,
    searchQueries: kgSearchQueries,
    allChains: chains,
    updatedVisitedEdges: newVisitedEdges,
  } = await knowledgeGraphModule.generateSearchChainsAndSearchQueries(
    rootQuery,
    outline,
    kg,
    llmModel,
    new Set<string>(),
    language,
    reportId,
    usageFile,
    true,
    historySearchQueries,
    cfg.kgQueryNum,
  );
  updateActionElapsedTime(
    "generate_explore_queries",
    reportId,
    usageFile,
    (Date.now() - genExploreStart) / 1000,
  );

  // Build next-round planned queries
  const planned = dedupPreserveOrder([
    ...(kgSearchQueries || []),
    ...outlineSearchQueries,
  ]);
  const executedSet = new Set(historySearchQueries);
  const nextSearchQueries = planned.filter((q) => !executedSet.has(q));

  return {
    outline,
    knowledgeGraphJson: serializeKg(kg),
    historySearchQueries,
    visitedUrls: [...visitedUrls],
    searchQueries: nextSearchQueries,
    visitedEdges: visitedEdgesArrayFromSet(newVisitedEdges),
    iterationIndex: 0,
    isTerminal: false,
    startTime: Date.now(),
  };
}

// ─── Node: iterate ───────────────────────────────────────────────────────────

async function iterateNode(
  state: ResearchStateType,
  llmModel: LLMModel,
): Promise<Partial<ResearchStateType>> {
  const {
    reportId,
    rootQuery,
    productDir,
    language,
    usageFile,
    maxIter,
    disableEarlyStopping,
    cfg,
    iterationIndex,
  } = state;

  if (iterationIndex >= maxIter) {
    return { isTerminal: true };
  }

  const iter = iterationIndex;
  let kg = deserializeKg(state.knowledgeGraphJson);
  let { searchQueries } = state;
  let historySearchQueries = [...state.historySearchQueries];
  const visitedUrls = new Set<string>(state.visitedUrls);
  let outline = state.outline;
  let visitedEdges = visitedEdgesSetFromArray(state.visitedEdges);

  // Step 1: Search
  const evidenceNodesBatch: EvidenceNode[][] = [];
  const searchStart = Date.now();
  for (const searchQuery of searchQueries) {
    const newEvidences = await searchModule.searchWithFilteringVisitedUrls(
      searchQuery,
      rootQuery,
      llmModel,
      language,
      visitedUrls,
      reportId,
      usageFile,
      cfg.searchProvider as "bing" | "serper",
    );
    const batch: EvidenceNode[] = [];
    for (const ev of newEvidences) {
      const node = addEvidenceNode(kg, ev.source_title, ev.source_url, ev.content);
      if (ev.source_url) visitedUrls.add(ev.source_url);
      batch.push(node);
    }
    evidenceNodesBatch.push(batch);
  }
  if (searchQueries.length > 0) {
    updateActionElapsedTime(
      "search_iter",
      reportId,
      usageFile,
      (Date.now() - searchStart) / 1000,
    );
  }
  historySearchQueries = dedupPreserveOrder([
    ...historySearchQueries,
    ...(searchQueries || []),
  ]);

  // Step 2: Update KG
  const updateKgStart = Date.now();
  kg = await knowledgeGraphModule.updateKnowledgeGraph(
    rootQuery,
    kg,
    evidenceNodesBatch,
    searchQueries,
    llmModel,
    reportId,
    usageFile,
  );
  updateActionElapsedTime(
    "update_knowledge_graph",
    reportId,
    usageFile,
    (Date.now() - updateKgStart) / 1000,
  );

  fs.writeFileSync(
    path.join(productDir, `knowledge_graph_case${reportId}_iter${iter + 1}.txt`),
    JSON.stringify(kg),
    "utf-8",
  );

  // Step 3: Update outline
  const updateOutlineStart = Date.now();
  const flatEvidences = evidenceNodesBatch.flat();
  outline = await outlineModule.updateOutlineByKg(
    rootQuery,
    outline,
    flatEvidences,
    kg,
    llmModel,
    language,
    reportId,
    usageFile,
  );
  updateActionElapsedTime(
    "update_outline_by_kg",
    reportId,
    usageFile,
    (Date.now() - updateOutlineStart) / 1000,
  );

  fs.writeFileSync(
    path.join(productDir, `outline_case${reportId}_iter${iter + 1}.txt`),
    outline,
    "utf-8",
  );

  // Step 4: Generate new search queries
  const genExploreStart = Date.now();
  const chainResult =
    await knowledgeGraphModule.generateSearchChainsAndSearchQueries(
      rootQuery,
      outline,
      kg,
      llmModel,
      visitedEdges,
      language,
      reportId,
      usageFile,
      true,
      historySearchQueries,
      cfg.kgQueryNum,
    );
  updateActionElapsedTime(
    "generate_explore_queries",
    reportId,
    usageFile,
    (Date.now() - genExploreStart) / 1000,
  );
  visitedEdges = chainResult.updatedVisitedEdges;

  const genSqStart = Date.now();
  const ogSearchQueries = await outlineModule.generateSearchQueries(
    outline,
    historySearchQueries,
    llmModel,
    language,
    {
      reportId,
      usageFile,
      isUseHistorySearchQueries: true,
      pendingSearchQueries: chainResult.searchQueries,
      queryNum: cfg.ogQueryNum,
    },
  );
  updateActionElapsedTime(
    "generate_search_queries",
    reportId,
    usageFile,
    (Date.now() - genSqStart) / 1000,
  );

  const planned = dedupPreserveOrder([
    ...(chainResult.searchQueries || []),
    ...(ogSearchQueries || []),
  ]);
  const executedSet = new Set(historySearchQueries);
  const nextSearchQueries = planned.filter((q) => !executedSet.has(q));

  // Step 5: Judge terminal
  const judgeStart = Date.now();
  const isTerminal = await judgeTerminalByOutline(
    rootQuery,
    outline,
    llmModel,
    {
      reportId,
      usageFile,
      disableEarlyStopping,
    },
  );
  updateActionElapsedTime(
    "judge_terminal",
    reportId,
    usageFile,
    (Date.now() - judgeStart) / 1000,
  );

  return {
    outline,
    knowledgeGraphJson: serializeKg(kg),
    historySearchQueries,
    visitedUrls: [...visitedUrls],
    searchQueries: nextSearchQueries,
    visitedEdges: visitedEdgesArrayFromSet(visitedEdges),
    iterationIndex: iter + 1,
    isTerminal,
  };
}

// ─── Node: writeReport ───────────────────────────────────────────────────────

async function writeReportNode(
  state: ResearchStateType,
  llmModel: LLMModel,
): Promise<Partial<ResearchStateType>> {
  const {
    reportId,
    rootQuery,
    reportDir,
    productDir,
    language,
    usageFile,
    outline,
  } = state;
  const kg = deserializeKg(state.knowledgeGraphJson);

  try {
    const writeStart = Date.now();
    const report = await writeModule.writeReportByOutlineKg(
      rootQuery,
      outline,
      kg,
      llmModel,
      language,
      reportId,
      usageFile,
      false,
    );
    updateActionElapsedTime(
      "write_report",
      reportId,
      usageFile,
      (Date.now() - writeStart) / 1000,
    );

    console.log(report);

    const reportPath = path.join(reportDir, `${reportId}.md`);
    if (!fs.existsSync(reportPath)) {
      fs.mkdirSync(reportDir, { recursive: true });
      fs.writeFileSync(reportPath, report, "utf-8");
      console.log(`[Done] Report ${reportId} generated and saved`);
    }

    const elapsedTime = (Date.now() - state.startTime) / 1000;
    updateElapsedTime(reportId, usageFile, elapsedTime);
    console.log(`[Timing] Report ${reportId} took ${elapsedTime.toFixed(2)}s`);

    return { report, success: true };
  } catch (e: unknown) {
    const errMsg = e instanceof Error ? e.message : String(e);
    // Save error artifacts
    atomicWriteText(
      path.join(reportDir, `error_report_outline_${reportId}.txt`),
      outline,
    );
    atomicWriteText(
      path.join(reportDir, `error_report_kg_${reportId}.txt`),
      state.knowledgeGraphJson,
    );
    atomicWriteText(
      path.join(reportDir, `error_report_${reportId}.txt`),
      `Error in writing report for ${reportId}. ${errMsg}`,
    );
    return { error: errMsg, success: false };
  }
}

// ─── Conditional edge ────────────────────────────────────────────────────────

function shouldContinueIterating(state: ResearchStateType): string {
  if (state.isTerminal || state.iterationIndex >= state.maxIter) {
    return "writeReport";
  }
  return "iterate";
}

// ─── Build the graph ─────────────────────────────────────────────────────────

export function buildResearchGraph(llmModel: LLMModel) {
  const graph = new StateGraph(ResearchState)
    .addNode("initialize", (state: ResearchStateType) =>
      initializeNode(state, llmModel),
    )
    .addNode("iterate", (state: ResearchStateType) =>
      iterateNode(state, llmModel),
    )
    .addNode("writeReport", (state: ResearchStateType) =>
      writeReportNode(state, llmModel),
    )
    .addEdge("__start__", "initialize")
    .addEdge("initialize", "iterate")
    .addConditionalEdges("iterate", shouldContinueIterating, [
      "iterate",
      "writeReport",
    ])
    .addEdge("writeReport", "__end__");

  return graph.compile();
}

// ─── Top-level invocation helper ─────────────────────────────────────────────

export interface ProcessReportOptions {
  reportId: number;
  rootQuery: string;
  reportDir: string;
  productDir: string;
  language: string;
  llmModel: LLMModel;
  usageFile: string;
  maxIter: number;
  cfg: RunConfig;
  disableEarlyStopping?: boolean;
  blockedUrls?: string[];
}

export async function processReport(
  opts: ProcessReportOptions,
): Promise<{ success: boolean; reportId: number; error?: string; skipped?: boolean }> {
  const reportPath = path.join(opts.reportDir, `${opts.reportId}.md`);
  if (fs.existsSync(reportPath)) {
    console.log(`[Skip] Report ${opts.reportId} already exists, skipping`);
    return { success: true, reportId: opts.reportId, skipped: true };
  }

  const app = buildResearchGraph(opts.llmModel);

  try {
    const result = await app.invoke({
      reportId: opts.reportId,
      rootQuery: opts.rootQuery,
      reportDir: opts.reportDir,
      productDir: opts.productDir,
      language: opts.language,
      usageFile: opts.usageFile,
      maxIter: opts.maxIter,
      disableEarlyStopping: opts.disableEarlyStopping ?? false,
      blockedUrls: opts.blockedUrls ?? [],
      cfg: opts.cfg,
      llmModelName: "",
      // Defaults for mutable state
      outline: "",
      knowledgeGraphJson: "{}",
      historySearchQueries: [],
      visitedUrls: [],
      searchQueries: [],
      visitedEdges: [],
      iterationIndex: 0,
      isTerminal: false,
      startTime: Date.now(),
      report: "",
      success: false,
      error: "",
    });

    return {
      success: result.success,
      reportId: opts.reportId,
      error: result.error || undefined,
    };
  } catch (e: unknown) {
    const errMsg = e instanceof Error ? e.message : String(e);
    console.error(
      `\n[Error] Exception while processing report ${opts.reportId}: ${errMsg}`,
    );
    atomicWriteText(
      path.join(opts.reportDir, `error_report_${opts.reportId}.txt`),
      `Error processing report ${opts.reportId}: ${errMsg}`,
    );
    return { success: false, reportId: opts.reportId, error: errMsg };
  }
}
