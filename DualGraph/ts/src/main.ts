#!/usr/bin/env node
/**
 * CLI entry point for DualGraph TypeScript port.
 * Mirrors the Python main.py CLI.
 */

import { Command } from "commander";
import * as dotenv from "dotenv";
import * as fs from "node:fs";
import * as path from "node:path";
import { AgentConfig, getLlmModel } from "./llmUtils.js";
import { processReport, type RunConfig } from "./graph.js";

// ─── Env helpers ─────────────────────────────────────────────────────────────

const BASE = path.resolve(
  path.dirname(new URL(import.meta.url).pathname),
  "..",
  "..",
  "deepresearch",
);

function envStr(name: string, fallback?: string): string | undefined {
  const v = process.env[name];
  if (v == null) return fallback;
  const cleaned = v.trim().replace(/^['"]|['"]$/g, "");
  return cleaned || fallback;
}

function envBool(name: string, fallback = false): boolean {
  const v = process.env[name];
  if (v == null) return fallback;
  return ["1", "true", "yes", "y", "on"].includes(
    v.trim().replace(/^['"]|['"]$/g, "").toLowerCase(),
  );
}

function envFloat(name: string, fallback: number): number {
  const v = process.env[name];
  if (v == null) return fallback;
  const n = parseFloat(v.trim().replace(/^['"]|['"]$/g, ""));
  return Number.isNaN(n) ? fallback : n;
}

// ─── LLM model factory ──────────────────────────────────────────────────────

function buildBaseUrl(): string {
  const raw = envStr("LLM_BASE_URL") ?? envStr("AZURE_OPENAI_ENDPOINT");
  let baseUrl = raw ?? "https://api.openai.com/v1";
  if (baseUrl.replace(/\/+$/, "").toLowerCase().endsWith("/openai")) {
    baseUrl = baseUrl.replace(/\/+$/, "").slice(0, -"/openai".length);
  }
  return baseUrl.replace(/\/+$/, "");
}

function createLlmModel(modelName: string) {
  const baseUrl = buildBaseUrl();
  let apiKey = envStr("LLM_API_KEY");
  if (apiKey === "") apiKey = undefined;
  const conf: AgentConfig = {
    llm_provider: envStr("LLM_PROVIDER", "openai") ?? "openai",
    llm_model_name: modelName,
    llm_api_key: apiKey,
    llm_base_url: baseUrl,
    llm_temperature: envFloat("LLM_TEMPERATURE", 0.0),
  };
  return getLlmModel(conf);
}

// ─── Dataset loading ─────────────────────────────────────────────────────────

interface DatasetBundle {
  selectedIds: number[];
  queries: string[];
}

function loadEvalDataset(
  datasetName: string,
  idRange: [number, number],
  reportDir: string,
): DatasetBundle {
  let selectedIds: number[] = [];
  for (let i = idRange[0]; i <= idRange[1]; i++) selectedIds.push(i);

  // Skip already-completed reports
  const existing = new Set<number>();
  if (fs.existsSync(reportDir)) {
    for (const f of fs.readdirSync(reportDir)) {
      if (f.endsWith(".md")) {
        const num = parseInt(f.replace(".md", ""), 10);
        if (!Number.isNaN(num)) existing.add(num);
      }
    }
  }
  selectedIds = selectedIds.filter((id) => !existing.has(id));

  // Load query.jsonl
  const jsonlPath = path.join(
    BASE,
    "..",
    "eval_dataset",
    datasetName,
    "query.jsonl",
  );
  if (!fs.existsSync(jsonlPath)) {
    throw new Error(
      `Dataset file not found: ${jsonlPath}\n` +
        `Place a query.jsonl in eval_dataset/${datasetName}/`,
    );
  }
  const lines = fs
    .readFileSync(jsonlPath, "utf-8")
    .split("\n")
    .filter((l) => l.trim());
  const queryDict = new Map<number, string>();
  for (const line of lines) {
    const obj = JSON.parse(line) as { id?: number; prompt?: string };
    if (obj.id != null && obj.prompt != null) {
      queryDict.set(obj.id, obj.prompt);
    }
  }

  const validIds: number[] = [];
  const queries: string[] = [];
  for (const id of selectedIds) {
    const q = queryDict.get(id);
    if (q != null) {
      validIds.push(id);
      queries.push(q);
    } else {
      console.warn(`[Warning] ID ${id} not found in query.jsonl, skipping`);
    }
  }

  console.log(`\n${"=".repeat(80)}`);
  console.log(`Dataset: ${datasetName}`);
  console.log(`  Existing reports: ${existing.size}`);
  console.log(`  To process: ${validIds.length}`);
  console.log(`  IDs: ${validIds}`);
  console.log(`${"=".repeat(80)}\n`);

  return { selectedIds: validIds, queries };
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

const program = new Command();
program
  .name("dualgraph")
  .description("DualGraph deep research pipeline (TypeScript + LangGraph)")
  .option("--models <names...>", "LLM model name(s)", ["gpt-4.1-20250414-2"])
  .option("--version <stamp>", "Version stamp for output dirs", "v1")
  .option(
    "--datasets <names...>",
    "Eval dataset name(s) (subfolder under eval_dataset/)",
    ["example"],
  )
  .option("--disable-early-stopping", "Disable early stopping", false)
  .option(
    "--kg-query-num <n>",
    "Queries from KG per iteration",
    (v: string) => parseInt(v, 10),
    10,
  )
  .option(
    "--og-query-num <n>",
    "Queries from OG per iteration",
    (v: string) => parseInt(v, 10),
    10,
  )
  .option(
    "--id-range <start> <end>",
    "Query ID range [start, end]",
    (v: string, prev: number[]) => {
      prev.push(parseInt(v, 10));
      return prev;
    },
    [] as number[],
  )
  .option(
    "--search-provider <provider>",
    "Search backend (bing/serper)",
    "serper",
  )
  .option(
    "--max-iter <n>",
    "Max research iterations per query",
    (v: string) => parseInt(v, 10),
    5,
  )
  .option("--language <lang>", "Report language (English/Chinese)", "English");

program.parse();
const opts = program.opts();

// Load .env
const envPath = path.join(BASE, "baselines", ".env");
if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
  console.log(`Loaded .env from ${envPath}`);
}

const idRange: [number, number] =
  opts.idRange.length >= 2 ? [opts.idRange[0], opts.idRange[1]] : [1, 1];
const disableEarlyStopping: boolean = opts.disableEarlyStopping;
const versionStamp: string = disableEarlyStopping
  ? `${opts.version}_NOES`
  : opts.version;

const cfg: RunConfig = {
  kgQueryNum: opts.kgQueryNum,
  ogQueryNum: opts.ogQueryNum,
  searchProvider: opts.searchProvider,
};

console.log(`[CLI] models=${opts.models}, version=${opts.version}`);
console.log(
  `[CLI] datasets=${opts.datasets}, early_stopping=${!disableEarlyStopping}, id_range=${idRange}`,
);

async function main() {
  for (const modelName of opts.models) {
    for (const datasetName of opts.datasets) {
      const maxIter: number = opts.maxIter;
      const saveDir = path.join(BASE, "..", "exp_saves");
      fs.mkdirSync(saveDir, { recursive: true });

      const reportDir = path.join(
        saveDir,
        `og_kg_reports_${modelName}_${versionStamp}_MAX_${maxIter}_${datasetName}`,
      );
      const productDir = path.join(
        saveDir,
        `og_kg_products_${modelName}_${versionStamp}_MAX_${maxIter}_${datasetName}`,
      );
      const usageFile = path.join(
        saveDir,
        `og_kg_usage_${modelName}_${versionStamp}_MAX_${maxIter}_${datasetName}.json`,
      );

      if (!fs.existsSync(usageFile)) {
        fs.writeFileSync(usageFile, "{}", "utf-8");
      }
      fs.mkdirSync(reportDir, { recursive: true });
      fs.mkdirSync(productDir, { recursive: true });

      const llmModel = createLlmModel(modelName);
      const ds = loadEvalDataset(datasetName, idRange, reportDir);

      if (ds.selectedIds.length === 0) {
        console.log("All IDs already processed, skipping.");
        continue;
      }

      let completed = 0;
      let failed = 0;
      let skipped = 0;

      // Process sequentially (TS is single-threaded; use workers for concurrency)
      for (let i = 0; i < ds.selectedIds.length; i++) {
        const reportId = ds.selectedIds[i];
        const rootQuery = ds.queries[i];
        const language: string = opts.language;

        const result = await processReport({
          reportId,
          rootQuery,
          reportDir,
          productDir,
          language,
          llmModel,
          usageFile,
          maxIter,
          cfg,
          disableEarlyStopping,
        });

        if (result.skipped) skipped++;
        else if (result.success) completed++;
        else failed++;

        console.log(
          `[${i + 1}/${ds.selectedIds.length}] Report ${reportId}: ${result.success ? "done" : "failed"}`,
        );
      }

      console.log(`\n${"=".repeat(80)}`);
      console.log("[Complete] All pending reports processed");
      console.log(`  - Succeeded: ${completed}`);
      console.log(`  - Skipped: ${skipped}`);
      console.log(`  - Failed: ${failed}`);
      console.log(`  - Total: ${ds.selectedIds.length}`);
      console.log(`${"=".repeat(80)}\n`);
    }
  }
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
