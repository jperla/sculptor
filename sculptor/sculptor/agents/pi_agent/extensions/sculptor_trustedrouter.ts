/**
 * Sculptor TrustedRouter provider extension.
 *
 * TrustedRouter is OpenRouter-compatible: the same request shapes
 * (`/v1/chat/completions`, Anthropic `/v1/messages`), the same model ids, and
 * the same `GET /v1/models` document. So this provider is pi's own OpenRouter
 * provider pointed at https://api.trustedrouter.com, not a second catalog:
 *
 * - Every model pi ships for `openrouter` that TrustedRouter also serves is
 *   registered from pi's built-in entry, with all of its tuned metadata (API,
 *   thinking levels, compatibility flags, mid-conversation effort, and so on),
 *   and only base URL, context window, output limit and price taken from
 *   TrustedRouter. This is what pi itself does when it narrows OpenRouter's
 *   catalog to a key (`personalizeOpenRouterCatalog`).
 * - Tool-capable models TrustedRouter serves that pi's OpenRouter catalog does
 *   not have are built the way pi builds OpenRouter's catalog
 *   (`buildOpenRouterCatalog`).
 * - pi infers some request behavior from an `openrouter.ai` URL (reasoning
 *   object, developer role, Anthropic cache markers, session affinity); those
 *   values are stated explicitly here, under any value the model already sets,
 *   so TrustedRouter requests are shaped exactly like OpenRouter's.
 *
 * Authentication is pi's own: a key saved through Sculptor's Settings lands in
 * `auth.json` under `trustedrouter` and takes precedence; otherwise the
 * `TRUSTEDROUTER_API_KEY` environment variable is used. No secret is read,
 * logged, or embedded here, and the catalog fetch sends no credentials.
 *
 * Fail-soft: if the catalog cannot be fetched or parsed, nothing is registered
 * and pi starts normally.
 *
 * Pinned with the pi binary as one immutable unit (PI_VERSION_RANGE in
 * `dependency_management_service`) and loaded explicitly via `-e`.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getBuiltinModels } from "@earendil-works/pi-ai/providers/all";

const PROVIDER_ID = "trustedrouter";
const OPENAI_BASE_URL = "https://api.trustedrouter.com/v1";
// The Anthropic Messages client appends /v1/messages, as with OpenRouter's https://openrouter.ai/api.
const ANTHROPIC_BASE_URL = "https://api.trustedrouter.com";
const MODELS_URL = `${OPENAI_BASE_URL}/models`;
const CATALOG_TIMEOUT_MS = 8000;

type Pricing = {
  prompt?: string;
  completion?: string;
  input_cache_read?: string;
  input_cache_write?: string;
  overrides?: Array<{
    min_prompt_tokens?: number;
    utc_start?: number;
    utc_end?: number;
    utc_days?: string[];
    prompt?: string;
    completion?: string;
    input_cache_read?: string;
    input_cache_write?: string;
  }>;
};

type CatalogRow = {
  id: string;
  name?: string;
  context_length?: number | null;
  top_provider?: { context_length?: number | null; max_completion_tokens?: number | null };
  pricing?: Pricing;
  supported_parameters?: string[];
  architecture?: { modality?: string };
  reasoning?: {
    mandatory?: boolean;
    supported_efforts?: string[];
  };
};

// pi's `openRouterCost`: $/token decimal strings to $/1M tokens; prompt-length overrides become tiers.
const perMillion = (value: string | undefined, fallback: number): number => {
  const perToken = Number.parseFloat(value ?? "");
  return perToken >= 0 ? Number((perToken * 1_000_000).toFixed(6)) : fallback;
};

export const openRouterCost = (pricing: Pricing | undefined) => {
  const base = {
    input: perMillion(pricing?.prompt, 0),
    output: perMillion(pricing?.completion, 0),
    cacheRead: perMillion(pricing?.input_cache_read, 0),
    cacheWrite: perMillion(pricing?.input_cache_write, 0),
  };
  const tiers = (pricing?.overrides ?? []).flatMap((override) =>
    override.min_prompt_tokens === undefined ||
    override.utc_start !== undefined ||
    override.utc_end !== undefined ||
    override.utc_days !== undefined
      ? []
      : [
          {
            inputTokensAbove: override.min_prompt_tokens,
            input: perMillion(override.prompt, base.input),
            output: perMillion(override.completion, base.output),
            cacheRead: perMillion(override.input_cache_read, base.cacheRead),
            cacheWrite: perMillion(override.input_cache_write, base.cacheWrite),
          },
        ],
  );
  return tiers.length > 0 ? { ...base, tiers } : base;
};

// pi's `getOpenRouterThinkingLevelMap`, through its `getEffortThinkingLevelMap`.
const THINKING_LEVELS = ["minimal", "low", "medium", "high", "xhigh", "max"] as const;
export const thinkingLevelMap = (reasoning: CatalogRow["reasoning"]) => {
  if (!reasoning) return undefined;
  const efforts = reasoning.supported_efforts ?? [];
  if (efforts.length === 0) return reasoning.mandatory === true ? { off: null } : undefined;
  const supported = new Set(efforts);
  if (!THINKING_LEVELS.some((level) => supported.has(level)) && !supported.has("none")) {
    return reasoning.mandatory === true ? { off: null } : undefined;
  }
  const map: Record<string, string | null> = {};
  for (const level of THINKING_LEVELS) map[level] = supported.has(level) ? level : null;
  return { ...map, off: reasoning.mandatory === true ? null : "none" };
};

// What pi infers for an `openrouter.ai` URL, stated for TrustedRouter's.
const openRouterRequestShape = (api: string, id: string) =>
  api === "anthropic-messages"
    ? { sendSessionAffinityHeaders: true, sessionAffinityFormat: "openrouter" }
    : {
        supportsDeveloperRole: id.startsWith("anthropic/") || id.startsWith("openai/"),
        thinkingFormat: "openrouter",
        sendSessionAffinityHeaders: true,
        sessionAffinityFormat: "openrouter",
        ...(/^~?anthropic\//.test(id) ? { cacheControlFormat: "anthropic" } : {}),
      };

const baseUrlFor = (api: string) => (api === "anthropic-messages" ? ANTHROPIC_BASE_URL : OPENAI_BASE_URL);

// pi's `buildOpenRouterCatalog`, for a model pi's OpenRouter catalog does not have.
const fromCatalogRow = (row: CatalogRow) => {
  const api = /^anthropic\//.test(row.id) && !row.id.endsWith(":batch") ? "anthropic-messages" : "openai-completions";
  const levels = thinkingLevelMap(row.reasoning);
  return {
    type: "chat",
    id: row.id,
    name: row.name ?? row.id,
    api,
    reasoning: row.supported_parameters?.includes("reasoning") || false,
    ...(levels && { thinkingLevelMap: levels }),
    input: row.architecture?.modality?.includes("image") ? ["text", "image"] : ["text"],
    cost: openRouterCost(row.pricing),
    contextWindow: row.top_provider?.context_length || row.context_length || 4096,
    maxTokens: row.top_provider?.max_completion_tokens || 4096,
  };
};

export const toPiModels = (catalog: unknown, openRouterModels: ReadonlyArray<Record<string, any>>) => {
  const rows = (catalog as { data?: unknown })?.data;
  if (!Array.isArray(rows)) return [];
  const served = new Map<string, CatalogRow>();
  for (const row of rows) {
    if (row && typeof row === "object" && typeof (row as CatalogRow).id === "string") {
      served.set((row as CatalogRow).id, row as CatalogRow);
    }
  }
  const models: Record<string, any>[] = [];
  const seen = new Set<string>();
  // pi's tuned OpenRouter entries, narrowed to what TrustedRouter serves (personalizeOpenRouterCatalog).
  for (const builtin of openRouterModels) {
    if (builtin.type !== "chat") continue;
    const row = served.get(builtin.id);
    if (!row) continue;
    const { provider: _provider, ...model } = builtin;
    models.push({
      ...model,
      baseUrl: baseUrlFor(model.api),
      contextWindow: row.top_provider?.context_length || row.context_length || model.contextWindow,
      maxTokens: row.top_provider?.max_completion_tokens || model.maxTokens,
      cost: row.pricing ? openRouterCost(row.pricing) : model.cost,
      compat: { ...openRouterRequestShape(model.api, model.id), ...model.compat },
    });
    seen.add(builtin.id);
  }
  // Tool-capable models only TrustedRouter serves, built as pi builds OpenRouter's catalog.
  for (const row of served.values()) {
    if (seen.has(row.id) || !row.supported_parameters?.includes("tools")) continue;
    const model = fromCatalogRow(row);
    models.push({ ...model, baseUrl: baseUrlFor(model.api), compat: openRouterRequestShape(model.api, model.id) });
  }
  return models;
};

export default async function sculptorTrustedRouterExtension(pi: ExtensionAPI): Promise<void> {
  let models: ReturnType<typeof toPiModels>;
  try {
    const response = await fetch(MODELS_URL, { signal: AbortSignal.timeout(CATALOG_TIMEOUT_MS) });
    if (!response.ok) return;
    models = toPiModels(await response.json(), getBuiltinModels("openrouter") as ReadonlyArray<Record<string, any>>);
  } catch {
    return;
  }
  if (models.length === 0) return;
  pi.registerProvider(PROVIDER_ID, {
    name: "TrustedRouter",
    baseUrl: OPENAI_BASE_URL,
    apiKey: "$TRUSTEDROUTER_API_KEY",
    models,
  } as Parameters<ExtensionAPI["registerProvider"]>[1]);
}
