/**
 * Terminal module – multi-criteria early stopping.
 * TypeScript port of terminal_module.py
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";
import type { LLMModel } from "./llmUtils.ts";
import { callLlmModel } from "./llmUtils.ts";
import { updateLlmUsage, safeJsonLoads } from "./utils.ts";

const PROMPT_LIB_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "prompt_lib",
);

function loadPrompt(name: string): Record<string, string> {
  const raw = fs.readFileSync(path.join(PROMPT_LIB_DIR, name), "utf-8");
  return yaml.load(raw) as Record<string, string>;
}

// ─── Judgement criteria (mirrors terminal_module.py) ─────────────────────────

const JUDGEMENT_CRITERIA = [
  {
    name: "Instruction following",
    description:
      "Evaluate how well the outline follows the user's instructions for an outline. This includes adherence to the specified topic and scope, intended audience, purpose, constraints, required sections, level of detail, tone, and any formatting or length requirements. The evaluation should also check outline-specific expectations such as a clear hierarchical structure (e.g., H1/H2/H3 or bullet levels), logical ordering, consistent granularity across sections, correct numbering if requested, and inclusion of all required components (e.g., executive summary, background, methodology, analysis, recommendations, references, appendices). Penalize missing required elements, inclusion of prohibited items, incorrect scope or level of detail, or deviation from the requested format.",
  },
  {
    name: "Depth",
    description:
      "Assess the comprehensiveness and analytical depth of the outline. High-depth outlines move beyond broad headings to include specific subpoints, key arguments, mechanisms or causal drivers, assumptions and uncertainties, methods to be used, metrics, and success criteria. They indicate sequencing and logic (what builds on what), note dependencies and open questions, and identify where evidence, examples, and visuals will be integrated. Shallow outlines list generic topics without meaningful substructure, rationale, or analytical scaffolding.",
  },
  {
    name: "Balance",
    description:
      "Evaluate the fairness and objectivity of the outline. Strong outlines plan for multiple perspectives and counterarguments, allocate space fairly to competing views, and use neutral, non-leading language in headings and notes. Where issues are controversial or multi-faceted, the outline should explicitly include sections for trade-offs, limitations, and counter-evidence. Poor outlines display bias, give disproportionate space to one side without justification, or omit salient opposing views.",
  },
  {
    name: "Breadth",
    description:
      "Evaluate how many distinct and relevant subtopics, perspectives, or contexts the outline covers, while staying focused on the brief. Excellent outlines include appropriate dimensions such as historical context, legal or regulatory considerations, economic or market factors, technical or operational aspects, ethical implications, social or cultural impacts, geographic or comparative analysis, stakeholder perspectives, risks and limitations, and implementation pathways. Coverage should be wide-ranging yet purposeful; simply presenting two sides of a debate is insufficient, and irrelevant tangents should be avoided.",
  },
  {
    name: "Support",
    description:
      "Evaluate the outline's evidentiary scaffolding and sourcing plan. Providing source URLs somewhere in the outline (for example, in a references section or via inline citations) is the minimum requirement; if no section provides source URLs, the score must be zero. Factual accuracy is necessary but not sufficient. Higher-quality outlines explicitly attribute planned factual claims to verifiable sources (such as peer-reviewed articles, government databases, or reputable news organizations) with traceable citations including author or outlet, date, and URL. Quantitative points specify concrete datasets or reports, time frames, and comparative benchmarks. Qualitative points identify concrete examples or case studies, clearly linked to the argument, with sources. Sources should be credible and balanced; cherry-picking or omission of clearly relevant counter-evidence is penalized. Original synthesis should build on cited material, not replace it.",
  },
  {
    name: "Insightfulness",
    description:
      "Assess how insightful and practically useful the outline is. Excellent outlines go beyond common templates by offering original structure or framing, highlighting non-obvious but relevant connections, and sequencing sections to surface key insights efficiently. Recommendations and proposed analyses are concrete and actionable, clearly indicating what will be done, where it will appear, and how outcomes will be measured. Strong outlines call out specific real-world examples or comparator cases (who did what, when, what outcomes were observed, and how they were measured) and propose suitable exhibits such as tables, charts, or frameworks with a clear analytical purpose. Vague, generic, or purely aspirational notes cannot score highly.",
  },
];

const PASS_THRESHOLDS_HIGH: Record<string, number> = {
  Depth: 9,
  Balance: 9,
  Breadth: 9,
  Support: 5,
  Insightfulness: 8,
  "Instruction following": 9,
};

const PASS_THRESHOLDS_LOW: Record<string, number> = {
  Depth: 8,
  Balance: 8,
  Breadth: 8,
  Support: 5,
  Insightfulness: 8,
  "Instruction following": 8,
};

function repairJson(text: string): unknown {
  const parsed = safeJsonLoads(text);
  if (parsed == null) {
    throw new SyntaxError("Failed to parse JSON from LLM response");
  }
  return parsed;
}

// ─── judgeTerminalByOutline ──────────────────────────────────────────────────

export async function judgeTerminalByOutline(
  rootQuery: string,
  outline: string,
  llmModel: LLMModel,
  options?: {
    reportId?: number;
    usageFile?: string;
    disableEarlyStopping?: boolean;
    thresholdLevel?: "high" | "low";
  },
): Promise<boolean> {
  if (options?.disableEarlyStopping) {
    return false;
  }

  const yamlData = loadPrompt("judge_terminal_by_outline.yaml");
  const systemPrompt = yamlData.system;

  const criteriaText = JUDGEMENT_CRITERIA.map(
    (c) => `- **${c.name}**: ${c.description}`,
  ).join("\n");
  const criteriaNames = JUDGEMENT_CRITERIA.map((c) => c.name);

  const userPrompt = yamlData.user
    .replace("{criteria_text}", criteriaText)
    .replace("{root_query}", rootQuery)
    .replace("{outline}", outline);

  const messages = [
    { role: "system" as const, content: systemPrompt },
    { role: "user" as const, content: userPrompt },
  ];

  const start = Date.now();
  let response = await callLlmModel(llmModel, messages, {
    temperature: 0.7,
  });
  const elapsed = (Date.now() - start) / 1000;

  if (options?.reportId !== undefined && options?.usageFile) {
    updateLlmUsage(
      response,
      "judge_terminal_by_outline",
      options.reportId,
      options.usageFile,
      response._call_elapsed_time || elapsed,
    );
  }

  // Parse response with retries
  const ratings: Record<string, number> = {};
  let parsed = false;

  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const responseJson = repairJson(response.content) as Record<
        string,
        { rating: number }
      >;
      for (const name of criteriaNames) {
        ratings[name] = Number(responseJson[name]?.rating ?? 0);
      }
      parsed = true;
      break;
    } catch {
      if (attempt < 3) {
        const retryStart = Date.now();
        response = await callLlmModel(llmModel, messages, {
          temperature: 0.7,
        });
        const retryElapsed = (Date.now() - retryStart) / 1000;
        if (options?.reportId !== undefined && options?.usageFile) {
          updateLlmUsage(
            response,
            "judge_terminal_by_outline",
            options.reportId,
            options.usageFile,
            response._call_elapsed_time || retryElapsed,
          );
        }
      }
    }
  }

  if (!parsed) {
    for (const name of criteriaNames) {
      ratings[name] = 0;
    }
  }

  const thresholds =
    options?.thresholdLevel === "high"
      ? PASS_THRESHOLDS_HIGH
      : PASS_THRESHOLDS_LOW;

  return Object.entries(thresholds).every(
    ([name, threshold]) => (ratings[name] ?? 0) >= threshold,
  );
}
