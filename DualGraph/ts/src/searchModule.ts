/**
 * Search module – web search, URL filtering, and evidence summarization.
 * TypeScript port of search_module.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import yaml from "js-yaml";
import type { LLMModel } from "./llmUtils.js";
import { callLlmModel } from "./llmUtils.js";
import {
  safeJsonLoads,
  updateFilterStats,
  updateLlmUsage,
  updateReadpageStats,
} from "./utils.js";

const PROMPT_LIB_DIR = path.resolve(
  path.dirname(new URL(import.meta.url).pathname),
  "prompt_lib",
);

function loadPrompt(name: string): Record<string, string> {
  const raw = fs.readFileSync(path.join(PROMPT_LIB_DIR, name), "utf-8");
  return yaml.load(raw) as Record<string, string>;
}

interface SearchResult {
  id: string;
  title: string;
  url: string;
  snippet: string;
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  count: number;
}

interface UrlSelectionResult {
  selected_urls?: unknown[];
}

interface SummarizeResult {
  rational?: string;
  evidence?: string;
  summary?: string;
  is_useful?: boolean;
}

export interface SearchEvidence {
  source_title: string;
  source_url: string;
  content: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return null;
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

function normalizeUrl(url: string): string {
  const trimmed = url.trim();
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  if (trimmed.startsWith("//")) {
    return `https:${trimmed}`;
  }
  return `https://${trimmed}`;
}

function normalizeLanguage(language: string): { language: string; country: string } {
  const lower = language.trim().toLowerCase();
  if (["english", "en"].includes(lower)) return { language: "en", country: "us" };
  if (["chinese", "中文", "zh", "cn"].includes(lower)) {
    return { language: "zh", country: "cn" };
  }
  return { language: "en", country: "us" };
}

function containsChinese(text: string): boolean {
  return /[\u4e00-\u9fff]/.test(text);
}

async function fetchJson(url: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  return (await response.json()) as unknown;
}

export async function _searchBing(
  query: string,
  numResults: number = 5,
  visitedUrls: Set<string> = new Set(),
): Promise<SearchResponse> {
  const appId = process.env.BING_APP_ID;
  const endpoint = process.env.BING_ENDPOINT;
  if (!appId || !endpoint) {
    throw new Error("BING_APP_ID and BING_ENDPOINT must be configured");
  }

  const params = new URLSearchParams({
    q: query,
    count: String(Math.min(Math.max(numResults * 2, numResults), 20)),
    appid: appId,
  });

  const raw = asRecord(await fetchJson(`${endpoint}?${params.toString()}`));
  const webPages = asRecord(raw?.webPages);
  const values = Array.isArray(webPages?.value) ? webPages.value : [];
  const results: SearchResult[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < values.length; i += 1) {
    const item = asRecord(values[i]);
    const url = typeof item?.url === "string" ? item.url : "";
    if (!url || visitedUrls.has(url) || seen.has(url)) continue;
    seen.add(url);
    results.push({
      id: `bing-${i}`,
      title: typeof item?.name === "string" ? item.name : "",
      url,
      snippet: typeof item?.snippet === "string" ? item.snippet : "",
    });
    if (results.length >= numResults) break;
  }

  return { query, results, count: results.length };
}

export async function _searchSerper(
  query: string,
  numResults: number = 5,
  visitedUrls: Set<string> = new Set(),
): Promise<SearchResponse> {
  const apiKey = process.env.SERPER_KEY_ID;
  if (!apiKey) {
    throw new Error("SERPER_KEY_ID must be configured");
  }

  const locale = containsChinese(query)
    ? { gl: "cn", hl: "zh-cn", location: "China" }
    : { gl: "us", hl: "en", location: "United States" };

  const raw = asRecord(
    await fetchJson("https://google.serper.dev/search", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-KEY": apiKey,
      },
      body: JSON.stringify({
        q: query,
        num: Math.min(Math.max(numResults * 2, numResults), 20),
        ...locale,
      }),
    }),
  );

  const organic = Array.isArray(raw?.organic) ? raw.organic : [];
  const results: SearchResult[] = [];
  const seen = new Set<string>();

  for (let i = 0; i < organic.length; i += 1) {
    const item = asRecord(organic[i]);
    const url = typeof item?.link === "string" ? item.link : "";
    if (!url || visitedUrls.has(url) || seen.has(url)) continue;
    seen.add(url);
    results.push({
      id: `serper-${i}`,
      title: typeof item?.title === "string" ? item.title : "",
      url,
      snippet: typeof item?.snippet === "string" ? item.snippet : "",
    });
    if (results.length >= numResults) break;
  }

  return { query, results, count: results.length };
}

async function selectUrlsToVisit(
  searchResults: SearchResult[],
  rootQuery: string,
  query: string,
  llmModel: LLMModel,
  language: string,
  reportId?: number,
  usageFile?: string,
): Promise<SearchResult[]> {
  if (searchResults.length === 0) return [];

  const yamlData = loadPrompt("select_url_to_visit.yaml");
  const searchResultsText = searchResults
    .map(
      (result, index) =>
        `${index + 1}. Title: ${result.title}\n   URL: ${result.url}\n   Snippet: ${result.snippet}`,
    )
    .join("\n\n");

  const response = await callLlmModel(
    llmModel,
    [
      { role: "system", content: yamlData.system ?? "" },
      {
        role: "user",
        content: `## Input\n### root query\n${rootQuery}\n\n### search query\n${query}\n\n### search results\n${searchResultsText}\n\nlanguage: ${language}`,
      },
    ],
    { temperature: 0.3, num_retry: 3 },
  );

  maybeTrackUsage(response, "select_urls_to_visit", reportId, usageFile);

  const parsed = asRecord(safeJsonLoads(response.content)) as UrlSelectionResult | null;
  const selected = Array.isArray(parsed?.selected_urls)
    ? parsed.selected_urls.filter(
        (value): value is string => typeof value === "string" && value.trim().length > 0,
      )
    : [];

  if (selected.length === 0) {
    return searchResults;
  }

  const selectedSet = new Set(selected);
  const filtered = searchResults.filter((result) => selectedSet.has(result.url));
  return filtered.length > 0 ? filtered : searchResults;
}

async function readPageByJina(url: string): Promise<string> {
  const readerUrl = process.env.READER_URL ?? "https://r.jina.ai";
  const authToken = process.env.JINA_API_KEYS?.split(",")[0]?.trim();
  const headers = authToken
    ? ({ Authorization: "Bearer " + authToken } as Record<string, string>)
    : undefined;
  const response = await fetch(`${readerUrl.replace(/\/$/, "")}/${normalizeUrl(url)}`, {
    headers,
  });
  if (!response.ok) {
    throw new Error(`Jina reader failed with ${response.status}`);
  }
  return await response.text();
}

async function readPageByFirecrawl(url: string): Promise<string> {
  const apiUrl = process.env.FIRECRAWL_API_URL;
  if (!apiUrl) {
    throw new Error("FIRECRAWL_API_URL is not configured");
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (process.env.FIRECRAWL_API_KEY) {
    headers.Authorization = "Bearer " + process.env.FIRECRAWL_API_KEY;
  }

  const response = await fetch(`${apiUrl.replace(/\/$/, "")}/v2/scrape`, {
    method: "POST",
    headers,
    body: JSON.stringify({ url: normalizeUrl(url), formats: ["markdown"] }),
  });
  if (!response.ok) {
    throw new Error(`Firecrawl failed with ${response.status}`);
  }

  const raw = asRecord((await response.json()) as unknown);
  const data = asRecord(raw?.data);
  const markdown =
    (typeof data?.markdown === "string" && data.markdown) ||
    (typeof raw?.markdown === "string" && raw.markdown) ||
    "";
  if (!markdown) {
    throw new Error("Firecrawl returned no markdown content");
  }
  return markdown;
}

async function readPageRaw(url: string): Promise<string> {
  const response = await fetch(normalizeUrl(url));
  if (!response.ok) {
    throw new Error(`Page fetch failed with ${response.status}`);
  }
  const html = await response.text();
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

async function readPage(url: string): Promise<{ content: string; method: string }> {
  const preferredMethod = (process.env.READPAGE_METHOD ?? "jina").toLowerCase();
  const methods = preferredMethod === "firecrawl" ? ["firecrawl", "jina", "raw"] : [preferredMethod, "jina", "raw"];

  for (const method of methods) {
    try {
      if (method === "firecrawl") {
        return { content: await readPageByFirecrawl(url), method };
      }
      if (method === "jina") {
        return { content: await readPageByJina(url), method };
      }
      if (method === "raw" || method === "fetch") {
        return { content: await readPageRaw(url), method: "raw" };
      }
    } catch {
      continue;
    }
  }

  throw new Error("Unable to read page content");
}

function buildSummarizePrompt(
  rootQuery: string,
  pageTitle: string,
  pageContent: string,
  language: string,
  searchGoal?: string,
): string {
  const lines = ["## Input", "### root query", rootQuery];
  if (searchGoal) {
    lines.push("", "### search goal", searchGoal);
  }
  lines.push("", "### web page title", pageTitle, "", "### web page content", pageContent, "", `language: ${language}`);
  return lines.join("\n");
}

export async function _fetchAndSummarize(
  url: string,
  rootQuery: string,
  llmModel: LLMModel,
  reportId?: number,
  usageFile?: string,
  options?: {
    searchGoal?: string;
    sourceTitle?: string;
    sourceSnippet?: string;
    language?: string;
  },
): Promise<SearchEvidence | null> {
  const language = options?.language ?? "en";
  let pageContent = options?.sourceSnippet?.trim() ?? "";
  let readMethod = "snippet";
  let readSuccess = false;

  try {
    const page = await readPage(url);
    pageContent = page.content;
    readMethod = page.method;
    readSuccess = true;
  } catch {
    readMethod = "snippet";
  }

  if (reportId !== undefined && usageFile) {
    updateReadpageStats(reportId, usageFile, readMethod, readSuccess);
  }

  const yamlData = loadPrompt("summarize_evidence.yaml");
  const response = await callLlmModel(
    llmModel,
    [
      { role: "system", content: yamlData.system_jina ?? yamlData.system ?? "" },
      {
        role: "user",
        content: buildSummarizePrompt(
          rootQuery,
          options?.sourceTitle ?? url,
          pageContent,
          language,
          options?.searchGoal,
        ),
      },
    ],
    { temperature: 0.7, num_retry: 3 },
  );

  maybeTrackUsage(response, "summarize_by_llm", reportId, usageFile);
  const parsed = asRecord(safeJsonLoads(response.content)) as SummarizeResult | null;
  const isUseful = parsed?.is_useful ?? false;
  if (!isUseful) return null;

  const summary = typeof parsed?.summary === "string" ? parsed.summary.trim() : "";
  const evidence = typeof parsed?.evidence === "string" ? parsed.evidence.trim() : "";

  return {
    source_title: options?.sourceTitle ?? url,
    source_url: normalizeUrl(url),
    content: `\n**Summary**: ${summary}\n**Evidence**: ${evidence}`.trim(),
  };
}

export async function searchWithFilteringVisitedUrls(
  query: string,
  rootQuery: string,
  llmModel: LLMModel,
  language: string,
  visitedUrls: Set<string> = new Set(),
  reportId?: number,
  usageFile?: string,
  searchProvider: "bing" | "serper" = "serper",
): Promise<SearchEvidence[]> {
  const locale = normalizeLanguage(language);
  const provider = searchProvider === "bing" ? _searchBing : _searchSerper;
  const searchResponse = await provider(query, 5, visitedUrls);
  const initialResults = searchResponse.results;

  const selectedResults = (await selectUrlsToVisit(
    initialResults,
    rootQuery,
    query,
    llmModel,
    locale.language,
    reportId,
    usageFile,
  )).slice(0, 5);

  if (reportId !== undefined && usageFile) {
    updateFilterStats(reportId, usageFile, "first", initialResults.length, selectedResults.length);
  }

  const evidences: SearchEvidence[] = [];
  for (const result of selectedResults) {
    const evidence = await _fetchAndSummarize(result.url, rootQuery, llmModel, reportId, usageFile, {
      searchGoal: query,
      sourceTitle: result.title,
      sourceSnippet: result.snippet,
      language: locale.language,
    });
    if (!evidence) continue;
    evidences.push(evidence);
    visitedUrls.add(evidence.source_url);
  }

  if (reportId !== undefined && usageFile) {
    updateFilterStats(reportId, usageFile, "second", selectedResults.length, evidences.length);
  }

  return evidences;
}
