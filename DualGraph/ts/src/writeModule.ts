/**
 * Write module – report generation and citation/reference cleanup.
 * TypeScript port of write_module.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import yaml from "js-yaml";
import type { EvidenceNode, KnowledgeGraph } from "./dataModel.ts";
import {
  getEvidenceNodeById,
  knowledgeGraphToTextForWriter,
} from "./dataModel.ts";
import type { LLMModel } from "./llmUtils.ts";
import { callLlmModel } from "./llmUtils.ts";
import { updateLlmUsage } from "./utils.ts";

const PROMPT_LIB_DIR = path.resolve(
  path.dirname(new URL(import.meta.url).pathname),
  "prompt_lib",
);

const REFERENCE_PATTERN_EN = "## References";
const REFERENCE_PATTERN_ZH = "## 参考文献";

function loadPrompt(name: string): Record<string, string> {
  const raw = fs.readFileSync(path.join(PROMPT_LIB_DIR, name), "utf-8");
  return yaml.load(raw) as Record<string, string>;
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

function parseOutlineLine(line: string): { cleanedLine: string; citations: Set<number> } {
  const citations = new Set<number>();
  const citationMatches = line.matchAll(/<citation>(.*?)<\/citation>/g);

  for (const match of citationMatches) {
    const rawIds = (match[1] ?? "").split(",");
    for (const rawId of rawIds) {
      const trimmed = rawId.trim();
      const parsed = Number.parseInt(trimmed.replace(/^id_/, ""), 10);
      if (Number.isFinite(parsed)) citations.add(parsed);
    }
  }

  return {
    cleanedLine: line.replace(/<citation>.*?<\/citation>/g, "").trim(),
    citations,
  };
}

function groupSections(outline: string): Array<{ sectionOutline: string; evidenceIds: Set<number> }> {
  const lines = outline
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length <= 1) return [];

  const sections: Array<{ sectionOutline: string; evidenceIds: Set<number> }> = [];
  let currentLines: string[] = [];
  let currentEvidenceIds = new Set<number>();

  for (const line of lines.slice(1)) {
    const isTopLevel = /^\d+\.\s/.test(line);
    const { cleanedLine, citations } = parseOutlineLine(line);

    if (isTopLevel && currentLines.length > 0) {
      sections.push({
        sectionOutline: currentLines.join("\n"),
        evidenceIds: currentEvidenceIds,
      });
      currentLines = [];
      currentEvidenceIds = new Set<number>();
    }

    currentLines.push(cleanedLine);
    for (const id of citations) currentEvidenceIds.add(id);
  }

  if (currentLines.length > 0) {
    sections.push({
      sectionOutline: currentLines.join("\n"),
      evidenceIds: currentEvidenceIds,
    });
  }

  return sections;
}

function buildEvidenceText(evidenceNodes: EvidenceNode[], useKgWriterPrompt: boolean): string {
  if (evidenceNodes.length === 0) return "";
  const lines = ["SUPPORTING EVIDENCE:"];
  for (const node of evidenceNodes) {
    if (node.id === null) continue;
    lines.push(`- ID: ${useKgWriterPrompt ? node.id : `id_${node.id}`}`);
    lines.push(`- Title: ${node.source_title}`);
    lines.push(`  Content: ${node.content}`);
    lines.push("");
  }
  return lines.join("\n").trim();
}

export function dedupAndRenumber(reportText: string, language: string): string {
  const cleanedText = reportText.replace(/id_/g, "");
  const headerRegex = /^\s*##\s*(References|参考文献)\s*:?\s*$/im;
  const headerMatch = headerRegex.exec(cleanedText);
  if (!headerMatch) return cleanedText.trim();

  const body = cleanedText.slice(0, headerMatch.index).trimEnd();
  const refsBody = cleanedText.slice(headerMatch.index + headerMatch[0].length).trim();
  const refLines = refsBody
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const match = /^\[(\d+)\]\s*(.+)$/.exec(line);
      if (!match) return null;
      const oldNum = Number.parseInt(match[1], 10);
      const content = match[2].trim();
      const urlMatch = content.match(/https?:\/\/[^\s\]]+/);
      const key = urlMatch?.[0] ?? content;
      return { oldNum, content, key };
    })
    .filter((line): line is { oldNum: number; content: string; key: string } => line !== null);

  if (refLines.length === 0) return body.trim();

  const seen = new Map<string, { newNum: number; content: string }>();
  const oldToNew = new Map<number, number>();
  for (const ref of refLines) {
    if (!seen.has(ref.key)) {
      seen.set(ref.key, { newNum: seen.size + 1, content: ref.content });
    }
    oldToNew.set(ref.oldNum, seen.get(ref.key)!.newNum);
  }

  const rewrittenBody = body.replace(/\[(\s*\d+(?:\s*,\s*\d+)*)\]/g, (_match, group) => {
    const newNums = [...new Set(
      String(group)
        .split(",")
        .map((value) => Number.parseInt(value.trim(), 10))
        .filter((value) => Number.isFinite(value) && oldToNew.has(value))
        .map((value) => oldToNew.get(value) as number),
    )].sort((a, b) => a - b);
    return newNums.length > 0 ? `[${newNums.join(", ")}]` : "";
  });

  const referencesHeader = ["en", "english"].includes(language.toLowerCase())
    ? REFERENCE_PATTERN_EN
    : REFERENCE_PATTERN_ZH;
  const rebuiltRefs = [...seen.values()].map(
    (ref) => `[${ref.newNum}] ${ref.content}`,
  );

  return `${rewrittenBody.trim()}\n\n${referencesHeader}:\n${rebuiltRefs.join("\n")}`;
}

export async function writeReportByOutlineKg(
  rootQuery: string,
  outline: string,
  knowledgeGraph: KnowledgeGraph,
  llmModel: LLMModel,
  language: string,
  reportId?: number,
  usageFile?: string,
  isUseKgToWriteReport: boolean = false,
): Promise<string> {
  if (!outline.trim()) {
    return `# Report for: ${rootQuery}\n\nNo outline provided.`;
  }

  const lines = outline
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
  const reportTitle = lines[0] ?? `Report on ${rootQuery}`;
  const sections = groupSections(outline);
  const promptName = isUseKgToWriteReport
    ? "write_by_outline_kg.yaml"
    : "write_by_outline.yaml";
  const systemPrompt = loadPrompt(promptName).system ?? "";

  let fullReport = `${reportTitle}\n\n`;

  for (const section of sections) {
    if (/[.]\s*(reference|参考文献|引用)/i.test(section.sectionOutline)) {
      continue;
    }

    const evidenceNodes = [...section.evidenceIds]
      .map((id) => getEvidenceNodeById(knowledgeGraph, id))
      .filter((node): node is EvidenceNode => node !== undefined);

    const evidenceText = buildEvidenceText(evidenceNodes, isUseKgToWriteReport);
    const userParts: string[] = [];

    if (isUseKgToWriteReport) {
      userParts.push(`Report Title: ${reportTitle}`);
      userParts.push(`Root Query: ${rootQuery}`);
      if (fullReport.trim()) userParts.push(`Previous Content:\n${fullReport}`);
      userParts.push(`Current Section Outline:\n${section.sectionOutline}`);
      userParts.push(
        `Relevant Knowledge Graph:\n${knowledgeGraphToTextForWriter(knowledgeGraph, section.evidenceIds)}`,
      );
      if (evidenceText) userParts.push(evidenceText);
      userParts.push(
        `Write the full content for this section. Integrate evidence naturally. Connect logically to previous sections. Output ONLY the section content. Language: ${language}`,
      );
    } else {
      userParts.push(`REPORT TITLE: ${reportTitle}`);
      userParts.push(`ROOT QUERY: ${rootQuery}`);
      if (fullReport.trim()) userParts.push(`PREVIOUS CONTENT:\n${fullReport}`);
      userParts.push(`CURRENT SECTION OUTLINE:\n${section.sectionOutline}`);
      if (evidenceText) userParts.push(evidenceText);
      userParts.push(
        `Write the full content for this section. Integrate evidence naturally. Connect logically to previous sections. Output ONLY the section content. Language: ${language}`,
      );
    }

    const response = await callLlmModel(
      llmModel,
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userParts.join("\n\n") },
      ],
      { temperature: 0.7, num_retry: 3 },
    );

    maybeTrackUsage(
      response,
      isUseKgToWriteReport ? "write_report_by_outline_kg" : "write_report_by_outline",
      reportId,
      usageFile,
    );
    fullReport += `${response.content.trim()}\n\n`;
  }

  const referencesHeader = ["en", "english"].includes(language.toLowerCase())
    ? REFERENCE_PATTERN_EN
    : REFERENCE_PATTERN_ZH;
  const references = knowledgeGraph.evidence_nodes
    .filter((node) => node.id !== null)
    .sort((a, b) => (a.id ?? 0) - (b.id ?? 0))
    .map((node) => `[${node.id}] ${node.source_title} - ${node.source_url}`);

  fullReport += `${referencesHeader}:\n${references.join("\n")}`;
  return dedupAndRenumber(fullReport, language);
}
