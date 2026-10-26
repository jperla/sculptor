/**
 * Sculptor TrustedRouter provider extension.
 *
 * TrustedRouter is OpenRouter-compatible: the same request shapes
 * (`/v1/chat/completions`, Anthropic `/v1/messages`, image output through
 * `modalities`), the same model ids, the same `GET /v1/models` and
 * `GET /v1/models/user` documents, and the same PKCE key exchange. So this
 * registers a native pi provider built from pi's own OpenRouter provider rather
 * than a second, hand-written integration:
 *
 * - Requests go through the OpenRouter provider's own stream and image
 *   implementations (`openrouterProvider().stream` / `generateImages`), which
 *   dispatch by model API and send to each model's base URL.
 * - Models are pi's built-in OpenRouter entries (chat and image) for every id
 *   TrustedRouter serves, with base URL, limits and price from TrustedRouter.
 *   Models only TrustedRouter serves are built as `buildOpenRouterCatalog`
 *   builds OpenRouter's.
 * - Request behavior pi infers from an `openrouter.ai` URL (reasoning object,
 *   developer role, Anthropic cache markers, session affinity) is stated
 *   explicitly, under any value the model already sets.
 * - With a key, `refreshModels` narrows the catalog to `GET /v1/models/user`,
 *   as pi does for OpenRouter (`personalizeOpenRouterCatalog`).
 * - Auth is an API key (`auth.json`, then `TRUSTEDROUTER_API_KEY`) or "Sign in
 *   with TrustedRouter", the same PKCE flow pi runs for OpenRouter.
 *
 * Not mirrored: OpenRouter's TypeSafe System One classifier models (TrustedRouter
 * serves decision models on its own `/api/alpha/decide` route instead).
 *
 * Fail-soft: if the catalog cannot be fetched or parsed, nothing is registered
 * and pi starts normally. No secret is logged or embedded here; the catalog
 * fetch sends no credentials.
 *
 * Pinned with the pi binary as one immutable unit (PI_VERSION_RANGE in
 * `dependency_management_service`) and loaded explicitly via `-e`.
 */
import { createServer } from "node:http";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  builtinProviders,
  getBuiltinImageModels,
  getBuiltinModels,
} from "@earendil-works/pi-ai/providers/all";

const PROVIDER_ID = "trustedrouter";
const PROVIDER_NAME = "TrustedRouter";
const OPENAI_BASE_URL = "https://api.trustedrouter.com/v1";
// The Anthropic Messages client appends /v1/messages, as with OpenRouter's https://openrouter.ai/api.
const ANTHROPIC_BASE_URL = "https://api.trustedrouter.com";
const AUTHORIZE_URL = "https://trustedrouter.com/auth";
const TOKEN_URL = "https://trustedrouter.com/v1/auth/keys";
const ENV_VAR = "TRUSTEDROUTER_API_KEY";
const CATALOG_TIMEOUT_MS = 8000;
const USER_MODELS_TIMEOUT_MS = 15_000;
const LOGIN_TIMEOUT_MS = 5 * 60 * 1000;
const TOKEN_EXCHANGE_TIMEOUT_MS = 30_000;

type AnyRecord = Record<string, any>;

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
  architecture?: { modality?: string; input_modalities?: string[]; output_modalities?: string[] };
  reasoning?: { mandatory?: boolean; supported_efforts?: string[] };
};

// ---------------------------------------------------------------------------
// Catalog: pi's OpenRouter conversions, applied to TrustedRouter's documents.
// ---------------------------------------------------------------------------

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
const openRouterRequestShape = (api: string, id: string): AnyRecord => {
  if (api === "anthropic-messages") return { sendSessionAffinityHeaders: true, sessionAffinityFormat: "openrouter" };
  if (api !== "openai-completions") return {};
  return {
    supportsDeveloperRole: id.startsWith("anthropic/") || id.startsWith("openai/"),
    thinkingFormat: "openrouter",
    sendSessionAffinityHeaders: true,
    sessionAffinityFormat: "openrouter",
    ...(/^~?anthropic\//.test(id) ? { cacheControlFormat: "anthropic" } : {}),
  };
};

const baseUrlFor = (api: string) => (api === "anthropic-messages" ? ANTHROPIC_BASE_URL : OPENAI_BASE_URL);

const modalities = (values: string[] | undefined): ("text" | "image")[] =>
  Array.from(new Set((values ?? []).filter((value): value is "text" | "image" => value === "text" || value === "image")));

const rowsById = (document: unknown): Map<string, CatalogRow> => {
  const rows = (document as { data?: unknown })?.data;
  const byId = new Map<string, CatalogRow>();
  if (!Array.isArray(rows)) return byId;
  for (const row of rows) {
    if (row && typeof row === "object" && typeof (row as CatalogRow).id === "string") byId.set((row as CatalogRow).id, row);
  }
  return byId;
};

// TrustedRouter's limits and price over a pi model (pi's `personalizeOpenRouterCatalog`).
const withServedLimits = (model: AnyRecord, row: CatalogRow): AnyRecord => ({
  ...model,
  ...(model.type === "image"
    ? {}
    : {
        contextWindow: row.top_provider?.context_length || row.context_length || model.contextWindow,
        maxTokens: row.top_provider?.max_completion_tokens || model.maxTokens,
      }),
  cost: row.pricing ? openRouterCost(row.pricing) : model.cost,
});

const asTrustedRouter = (model: AnyRecord): AnyRecord => ({
  ...model,
  provider: PROVIDER_ID,
  baseUrl: model.type === "image" ? OPENAI_BASE_URL : baseUrlFor(model.api),
  ...(model.type === "image" ? {} : { compat: { ...openRouterRequestShape(model.api, model.id), ...model.compat } }),
});

// pi's `buildOpenRouterCatalog` chat entry, for a model pi's OpenRouter catalog does not have.
const chatFromRow = (row: CatalogRow): AnyRecord => {
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

// pi's `buildOpenRouterCatalog` image entry.
const imageFromRow = (row: CatalogRow): AnyRecord | undefined => {
  const output = modalities(row.architecture?.output_modalities);
  if (!output.includes("image")) return undefined;
  const input = modalities(row.architecture?.input_modalities);
  return {
    type: "image",
    id: row.id,
    name: row.name ?? row.id,
    api: "openrouter-images",
    input: input.length > 0 ? input : ["text"],
    output,
    cost: openRouterCost(row.pricing),
  };
};

export const buildCatalog = (
  chatDocument: unknown,
  imageDocument: unknown,
  openRouterChat: ReadonlyArray<AnyRecord>,
  openRouterImages: ReadonlyArray<AnyRecord>,
): AnyRecord[] => {
  const servedChat = rowsById(chatDocument);
  const servedImages = rowsById(imageDocument);
  const models: AnyRecord[] = [];
  const seen = new Set<string>();
  for (const builtin of openRouterChat) {
    const row = servedChat.get(builtin.id);
    if (builtin.type !== "chat" || !row) continue;
    models.push(asTrustedRouter(withServedLimits(builtin, row)));
    seen.add(`chat:${builtin.id}`);
  }
  for (const row of servedChat.values()) {
    if (seen.has(`chat:${row.id}`) || !row.supported_parameters?.includes("tools")) continue;
    models.push(asTrustedRouter(chatFromRow(row)));
  }
  for (const builtin of openRouterImages) {
    const row = servedImages.get(builtin.id);
    if (builtin.type !== "image" || !row) continue;
    models.push(asTrustedRouter(withServedLimits(builtin, row)));
    seen.add(`image:${builtin.id}`);
  }
  for (const row of servedImages.values()) {
    if (seen.has(`image:${row.id}`)) continue;
    const model = imageFromRow(row);
    if (model) models.push(asTrustedRouter(model));
  }
  return models;
};

// Narrow to the key's models, limits and prices (pi's `personalizeOpenRouterCatalog`).
export const personalize = (catalog: ReadonlyArray<AnyRecord>, userDocument: unknown): AnyRecord[] => {
  const allowed = rowsById(userDocument);
  return catalog.flatMap((model) => {
    if (model.type !== "chat") return [model];
    const row = allowed.get(model.id);
    return row ? [withServedLimits(model, row)] : [];
  });
};

// ---------------------------------------------------------------------------
// Auth: an API key, or "Sign in with TrustedRouter" (pi's OpenRouter PKCE flow).
// ---------------------------------------------------------------------------

const apiKeyAuth = {
  name: `${PROVIDER_NAME} API key`,
  login: async (interaction: AnyRecord) => {
    interaction.signal.throwIfAborted();
    const key = await interaction.prompt({ type: "secret", message: `Enter ${PROVIDER_NAME} API key` });
    interaction.signal.throwIfAborted();
    return { type: "api_key", key };
  },
  resolve: async ({ ctx, credential, signal }: AnyRecord) => {
    signal.throwIfAborted();
    if (credential?.key) return { auth: { apiKey: credential.key }, env: credential.env, source: "stored credential" };
    const value = await ctx.env(ENV_VAR);
    signal.throwIfAborted();
    return value ? { auth: { apiKey: value }, source: ENV_VAR } : undefined;
  },
};

const base64url = (bytes: Uint8Array): string =>
  btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=/g, "");

const generatePKCE = async () => {
  const verifier = base64url(crypto.getRandomValues(new Uint8Array(32)));
  const challenge = base64url(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier))));
  return { verifier, challenge };
};

const page = (message: string) =>
  `<!doctype html><meta charset="utf-8"><title>${PROVIDER_NAME}</title><p style="font-family:system-ui;margin:3rem">${message}</p>`;

const exchangeAuthorizationCode = async (code: string, verifier: string, signal: AbortSignal) => {
  const response = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { accept: "application/json", "content-type": "application/json" },
    body: JSON.stringify({ code, code_verifier: verifier, code_challenge_method: "S256" }),
    signal: AbortSignal.any([signal, AbortSignal.timeout(TOKEN_EXCHANGE_TIMEOUT_MS)]),
  });
  let body: AnyRecord = {};
  try {
    body = (await response.json()) ?? {};
  } catch {
    if (response.ok) throw new Error(`${PROVIDER_NAME} OAuth returned invalid JSON`);
  }
  if (!response.ok) {
    const detail = body?.error?.message ?? body?.message ?? body?.error_description;
    throw new Error(`${PROVIDER_NAME} OAuth key exchange failed (HTTP ${response.status})${detail ? `: ${detail}` : ""}`);
  }
  if (typeof body.key !== "string" || body.key.length === 0) throw new Error(`${PROVIDER_NAME} OAuth response carries no "key"`);
  return { type: "oauth", access: body.key, refresh: "", expires: Number.MAX_SAFE_INTEGER };
};

const parseAuthorizationInput = (input: string): string | undefined => {
  const value = input.trim();
  if (!value) return undefined;
  try {
    return new URL(value).searchParams.get("code") ?? undefined;
  } catch {
    // Not a URL: a bare code or a query string.
  }
  return value.includes("code=") ? (new URLSearchParams(value).get("code") ?? undefined) : value;
};

const login = async (interaction: AnyRecord) => {
  const { verifier, challenge } = await generatePKCE();
  const path = `/oauth/callback/${crypto.randomUUID()}`;
  let settle: (value: AnyRecord | undefined, error?: Error) => void = () => {};
  const callbackResult = new Promise<AnyRecord | undefined>((resolve, reject) => {
    settle = (value, error) => (error ? reject(error) : resolve(value));
  });
  callbackResult.catch(() => undefined);
  let handled = false;
  const server = createServer((request, response) => {
    void (async () => {
      const url = new URL(request.url ?? "/", "http://localhost");
      const send = (status: number, message: string) => {
        response.writeHead(status, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
        response.end(page(message));
      };
      if (request.method !== "GET" || url.pathname !== path) return send(404, "Callback route not found.");
      if (handled) return send(409, "This sign-in has already been handled.");
      const error = url.searchParams.get("error");
      if (error) {
        send(400, `${PROVIDER_NAME} authorization failed.`);
        return settle(undefined, new Error(`${PROVIDER_NAME} authorization failed: ${url.searchParams.get("error_description") ?? error}`));
      }
      const code = url.searchParams.get("code");
      if (!code) return send(400, "Missing authorization code.");
      handled = true;
      try {
        const credential = await exchangeAuthorizationCode(code, verifier, interaction.signal);
        send(200, `Signed in to ${PROVIDER_NAME}. You may now close this page.`);
        settle(credential);
      } catch (failure) {
        send(502, `${PROVIDER_NAME} sign-in failed.`);
        settle(undefined, failure instanceof Error ? failure : new Error(String(failure)));
      }
    })();
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, process.env.PI_OAUTH_CALLBACK_HOST || "127.0.0.1", () => resolve());
  });
  const timer = setTimeout(() => settle(undefined, new Error(`${PROVIDER_NAME} sign-in timed out`)), LOGIN_TIMEOUT_MS);
  const manualAbort = new AbortController();
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("OAuth callback server did not bind to TCP");
    const host = address.address.includes(":") ? `[${address.address}]` : address.address;
    const redirectUri = `http://${host}:${address.port}${path}`;
    const authorizeUrl = new URL(AUTHORIZE_URL);
    authorizeUrl.search = new URLSearchParams({
      callback_url: redirectUri,
      code_challenge: challenge,
      code_challenge_method: "S256",
    }).toString();
    interaction.notify({ type: "progress", message: `Listening for ${PROVIDER_NAME} OAuth callback on ${redirectUri}` });
    interaction.notify({
      type: "auth_url",
      url: authorizeUrl.toString(),
      instructions: "Complete sign-in in your browser. If the browser is on another machine, paste the final redirect URL here.",
    });
    // As in pi's waitForCallbackOrManualInput: whichever of the callback or a pasted code comes first.
    const manual = interaction
      .prompt({
        type: "manual_code",
        message: "Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
        placeholder: redirectUri,
        signal: manualAbort.signal,
      })
      .then((input: string) => ({ input }))
      .catch((error: unknown) => ({ error }));
    const first = await Promise.race([callbackResult.then((credential) => ({ credential })), manual]);
    if ("credential" in first && first.credential) return first.credential;
    if ("error" in first) throw first.error instanceof Error ? first.error : new Error(String(first.error));
    const code = "input" in first ? parseAuthorizationInput(first.input ?? "") : undefined;
    if (!code) throw new Error("Missing authorization code");
    interaction.notify({ type: "progress", message: "Exchanging authorization code for an API key..." });
    return await exchangeAuthorizationCode(code, verifier, interaction.signal);
  } finally {
    clearTimeout(timer);
    manualAbort.abort();
    server.close();
  }
};

const oauth = {
  name: `${PROVIDER_NAME} OAuth`,
  loginLabel: `Sign in with ${PROVIDER_NAME}`,
  login,
  refresh: async (credential: AnyRecord) => credential,
  toAuth: async (credential: AnyRecord) => ({ apiKey: credential.access }),
};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const fetchJson = async (url: string, timeoutMs: number, signal?: AbortSignal, key?: string) => {
  const response = await fetch(url, {
    headers: { accept: "application/json", ...(key ? { authorization: `Bearer ${key}` } : {}) },
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs),
  });
  return response;
};

export default async function sculptorTrustedRouterExtension(pi: ExtensionAPI): Promise<void> {
  const openrouter = builtinProviders().find((provider) => provider.id === "openrouter") as AnyRecord | undefined;
  if (!openrouter) return;
  let catalog: AnyRecord[];
  try {
    const [chatResponse, imageResponse] = await Promise.all([
      fetchJson(`${OPENAI_BASE_URL}/models`, CATALOG_TIMEOUT_MS),
      fetchJson(`${OPENAI_BASE_URL}/models?output_modalities=image`, CATALOG_TIMEOUT_MS),
    ]);
    if (!chatResponse.ok) return;
    catalog = buildCatalog(
      await chatResponse.json(),
      imageResponse.ok ? await imageResponse.json() : null,
      getBuiltinModels("openrouter") as ReadonlyArray<AnyRecord>,
      getBuiltinImageModels("openrouter") as ReadonlyArray<AnyRecord>,
    );
  } catch {
    return;
  }
  if (!catalog.some((model) => model.type === "chat")) return;

  let personalModels: AnyRecord[] = [];
  const allModels = () => (personalModels.length > 0 ? personalModels : catalog);

  const provider: AnyRecord = {
    id: PROVIDER_ID,
    name: PROVIDER_NAME,
    baseUrl: OPENAI_BASE_URL,
    auth: { apiKey: apiKeyAuth, oauth },
    getModels: () => allModels().filter((model) => model.type === "chat"),
    getAllModels: allModels,
    // pi's `withPersonalizedCatalog`, with TrustedRouter's /v1/models/user.
    refreshModels: async (context: AnyRecord) => {
      const restored = (context.stored?.models ?? []).filter((model: AnyRecord) => model.provider === PROVIDER_ID);
      if (!(await context.publish({ update: () => { personalModels = restored; } }))) return;
      if (!context.allowNetwork || context.signal.aborted) return;
      const credential = context.credential;
      const key = credential?.type === "oauth" ? credential.access : credential?.key;
      if (!key) return;
      const base = (context.baseUrl ?? OPENAI_BASE_URL).replace(/\/+$/, "");
      const response = await fetchJson(`${base}/models/user`, USER_MODELS_TIMEOUT_MS, context.signal, key);
      if ([401, 403, 404].includes(response.status)) return;
      if (!response.ok) throw new Error(`${PROVIDER_NAME} model list request failed: ${response.status}`);
      const personalized = personalize(catalog, await response.json());
      if (context.signal.aborted) return;
      await context.publish({
        persist: { models: personalized, checkedAt: Date.now() },
        update: () => { personalModels = personalized; },
      });
    },
    stream: (model: AnyRecord, context: AnyRecord, options: AnyRecord) => openrouter.stream(model, context, options),
    streamSimple: (model: AnyRecord, context: AnyRecord, options: AnyRecord) => openrouter.streamSimple(model, context, options),
    ...(openrouter.generateImages
      ? { generateImages: (model: AnyRecord, context: AnyRecord, options: AnyRecord) => openrouter.generateImages(model, context, options) }
      : {}),
  };
  pi.registerProvider(provider as Parameters<ExtensionAPI["registerProvider"]>[0]);
}
