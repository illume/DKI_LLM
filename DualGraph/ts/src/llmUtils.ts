/**
 * LLM utilities – wraps LLM calls via LangChain with retry logic.
 * TypeScript port of llm_utils.py
 */

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { SystemMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import type { BaseMessage } from "@langchain/core/messages";

// ─── ChatMessage ─────────────────────────────────────────────────────────────
// Simple message type used by consumer modules (replaces OpenAI's type).

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

function toLangChainMessages(messages: ChatMessage[]): BaseMessage[] {
  return messages.map((m) => {
    switch (m.role) {
      case "system":
        return new SystemMessage(m.content);
      case "user":
        return new HumanMessage(m.content);
      case "assistant":
        return new AIMessage(m.content);
    }
  });
}

// ─── ModelResponse ───────────────────────────────────────────────────────────

export interface ModelResponse {
  content: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  _call_elapsed_time: number;
}

function makeModelResponse(partial?: Partial<ModelResponse>): ModelResponse {
  return {
    content: partial?.content ?? "",
    usage: partial?.usage ?? {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0,
    },
    _call_elapsed_time: partial?._call_elapsed_time ?? 0,
  };
}

// ─── AgentConfig ─────────────────────────────────────────────────────────────

export interface AgentConfig {
  llm_provider: string;
  llm_model_name?: string;
  llm_api_key?: string;
  llm_base_url?: string;
  llm_temperature: number;
  llm_api_version?: string;
  llm_extra_headers?: Record<string, string>;
}

export function makeAgentConfig(
  partial?: Partial<AgentConfig>,
): AgentConfig {
  return {
    llm_provider: partial?.llm_provider ?? "openai",
    llm_model_name: partial?.llm_model_name,
    llm_api_key: partial?.llm_api_key,
    llm_base_url: partial?.llm_base_url,
    llm_temperature: partial?.llm_temperature ?? 0.0,
    llm_api_version: partial?.llm_api_version,
    llm_extra_headers: partial?.llm_extra_headers,
  };
}

// ─── LLMModel ────────────────────────────────────────────────────────────────

export class LLMModel {
  chatModel: ChatOpenAI;
  modelName: string;

  constructor(chatModel: ChatOpenAI, modelName: string) {
    this.chatModel = chatModel;
    this.modelName = modelName;
  }

  async completion(
    messages: ChatMessage[],
    options?: {
      temperature?: number;
      max_tokens?: number;
      stop?: string[];
      model_name?: string;
    },
  ): Promise<ModelResponse> {
    let model = this.chatModel;

    // Apply per-call overrides via .bind()
    const bindArgs: Record<string, unknown> = {};
    if (options?.model_name) bindArgs.model = options.model_name;
    if (options?.temperature !== undefined)
      bindArgs.temperature = options.temperature;
    if (options?.max_tokens !== undefined)
      bindArgs.max_tokens = options.max_tokens;
    if (options?.stop !== undefined) bindArgs.stop = options.stop;
    if (Object.keys(bindArgs).length > 0) {
      model = model.bind(bindArgs) as unknown as ChatOpenAI;
    }

    const langchainMessages = toLangChainMessages(messages);
    const result = await model.invoke(langchainMessages);

    const content =
      typeof result.content === "string"
        ? result.content
        : JSON.stringify(result.content);

    const tokenUsage = result.usage_metadata;
    return makeModelResponse({
      content,
      usage: {
        prompt_tokens: tokenUsage?.input_tokens ?? 0,
        completion_tokens: tokenUsage?.output_tokens ?? 0,
        total_tokens: tokenUsage?.total_tokens ?? 0,
      },
    });
  }
}

// ─── getLlmModel ─────────────────────────────────────────────────────────────

export function getLlmModel(conf?: AgentConfig): LLMModel {
  const c = conf ?? makeAgentConfig({
    llm_provider: process.env.LLM_PROVIDER ?? "openai",
    llm_base_url: process.env.LLM_BASE_URL,
    llm_api_key: process.env.LLM_API_KEY,
  });

  const provider = c.llm_provider || "openai";
  const apiKey =
    c.llm_api_key ?? process.env.OPENAI_API_KEY ?? "";
  const baseURL =
    c.llm_base_url ??
    process.env.OPENAI_ENDPOINT ??
    "https://api.openai.com/v1";

  const modelName = c.llm_model_name ?? "";

  if (provider === "custom") {
    const resolvedKey = apiKey || process.env.LLM_API_KEY || "EMPTY";
    const resolvedUrl = c.llm_base_url || process.env.LLM_BASE_URL;

    const chatModelOpts: ConstructorParameters<typeof ChatOpenAI>[0] = {
      openAIApiKey: resolvedKey,
      modelName,
      timeout: 180_000,
      configuration: {},
    };
    if (resolvedUrl) {
      chatModelOpts.configuration = {
        ...chatModelOpts.configuration,
        baseURL: resolvedUrl,
      };
    }
    if (c.llm_extra_headers) {
      chatModelOpts.configuration = {
        ...chatModelOpts.configuration,
        defaultHeaders: c.llm_extra_headers,
      };
    }
    const chatModel = new ChatOpenAI(chatModelOpts);
    return new LLMModel(chatModel, modelName);
  }

  // Default: generic OpenAI-compatible
  const chatModel = new ChatOpenAI({
    openAIApiKey: apiKey || "EMPTY",
    modelName,
    timeout: 180_000,
    configuration: {
      baseURL,
    },
  });
  return new LLMModel(chatModel, modelName);
}

// ─── callLlmModel (with retry) ──────────────────────────────────────────────

const RETRY_AFTER_RE = /retry after\s+(\d+(?:\.\d+)?)\s+second/i;

function isRateLimitError(err: Error): boolean {
  const s = String(err).toLowerCase();
  return (
    s.includes("ratelimitreached") ||
    s.includes("rate limit") ||
    s.includes("error code: 429") ||
    s.includes("status code: 429") ||
    s.includes("http 429")
  );
}

function extractRetryAfter(err: Error): number | null {
  const m = RETRY_AFTER_RE.exec(String(err));
  if (!m) return null;
  return parseFloat(m[1]);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function callLlmModel(
  llmModel: LLMModel,
  messages: ChatMessage[],
  options?: {
    temperature?: number;
    max_tokens?: number;
    stop?: string[];
    num_retry?: number;
    rate_limit_backoff_base?: number;
    rate_limit_backoff_cap?: number;
  },
): Promise<ModelResponse> {
  const numRetry = options?.num_retry ?? 5;
  const backoffBase = options?.rate_limit_backoff_base ?? 1.0;
  const backoffCap = options?.rate_limit_backoff_cap ?? 30.0;

  for (let attempt = 0; attempt < numRetry; attempt++) {
    try {
      const callStart = Date.now();
      const response = await llmModel.completion(messages, {
        temperature: options?.temperature,
        max_tokens: options?.max_tokens,
        stop: options?.stop,
      });
      const elapsed = (Date.now() - callStart) / 1000;

      if (response?.content?.trim()) {
        response._call_elapsed_time = elapsed;
        return response;
      } else {
        console.warn(
          `LLM returned empty content on attempt ${attempt + 1}/${numRetry}`,
        );
      }
    } catch (e: unknown) {
      const err = e instanceof Error ? e : new Error(String(e));
      console.warn(
        `LLM call failed on attempt ${attempt + 1}/${numRetry}: ${err.message}`,
      );
      if (isRateLimitError(err) && attempt < numRetry - 1) {
        const retryAfter = extractRetryAfter(err);
        const high = Math.min(backoffCap, backoffBase * 2 ** attempt);
        const low = retryAfter !== null ? Math.max(0, retryAfter) : 0;
        const cappedHigh = Math.max(high, low);
        const sleepS = low + Math.random() * (cappedHigh - low);
        console.warn(
          `Rate-limit (429). Backing off ${sleepS.toFixed(2)}s before retry.`,
        );
        await sleep(sleepS * 1000);
      }
      if (attempt === numRetry - 1) throw err;
    }
  }

  throw new Error("All LLM retries exhausted with empty responses");
}

// ─── embedTexts ──────────────────────────────────────────────────────────────

export async function embedTexts(
  texts: string[],
  embeddingModel: string = "text-embedding-3-large",
  batchSize: number = 96,
  conf?: AgentConfig,
): Promise<number[][]> {
  if (texts.length === 0) return [];

  const c = conf ?? makeAgentConfig({
    llm_provider: process.env.LLM_PROVIDER ?? "openai",
    llm_base_url: process.env.LLM_BASE_URL,
    llm_api_key: process.env.LLM_API_KEY,
  });

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: c.llm_api_key ?? process.env.OPENAI_API_KEY ?? "EMPTY",
    modelName: embeddingModel,
    batchSize,
    timeout: 180_000,
    configuration: {
      baseURL:
        c.llm_base_url ??
        process.env.OPENAI_ENDPOINT ??
        "https://api.openai.com/v1",
    },
  });

  return embeddings.embedDocuments(texts);
}
