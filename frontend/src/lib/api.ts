// Typed fetch layer over the FastAPI backend. Base is the relative "/api" prefix so the
// same code works in dev (Vite proxies /api → :8001) and behind a single origin in prod.
import { getToken } from "./auth";

const BASE = import.meta.env.VITE_API_BASE || "/api";
// If a production static host has no API URL configured, present a browser-only
// synthetic demo instead of sending requests to that host's /api route.
export const IS_STATIC_DEMO = import.meta.env.PROD && !import.meta.env.VITE_API_BASE;

// Fields are declared, not constructor parameter properties: tsconfig sets
// erasableSyntaxOnly, which rejects `constructor(readonly status: number)`.
export class ApiError extends Error {
  status: number;
  body: unknown;

  constructor(status: number, body: unknown) {
    super(`request failed with ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

const demoPeople = [
  makeDemoPerson("demo-person-001", "Synthetic Profile 001", "High", 0.79, "Heavy workload", "Wellbeing check-in recommended"),
  makeDemoPerson("demo-person-002", "Synthetic Profile 002", "Moderate", 0.56, "Rotating shifts", "Monitor rest and recovery"),
  makeDemoPerson("demo-person-003", "Synthetic Profile 003", "Low", 0.18, "Stable workload", "No immediate follow-up indicated"),
];

function makeDemoPerson(id: string, name: string, band: "Low" | "Moderate" | "High", probability: number, workload: string, wellness: string) {
  const raw_records = Array.from({ length: 4 }, (_, index) => ({
    personnel_id: id,
    week: index + 1,
    years_of_service: 6,
    hardship_posting_flag: 0,
    transfer_count_24mo: 1,
    years_in_current_posting: 2,
    weekly_duty_hours: band === "High" ? 66 - index : band === "Moderate" ? 56 : 44,
    night_shift_ratio: band === "High" ? 0.55 : band === "Moderate" ? 0.3 : 0.12,
    overtime_hours: band === "High" ? 17 - index : band === "Moderate" ? 9 : 2,
    days_since_last_rest: band === "High" ? 11 - index : band === "Moderate" ? 7 : 3,
    days_since_last_leave: band === "High" ? 104 : band === "Moderate" ? 68 : 32,
    leave_balance: band === "High" ? 9 : band === "Moderate" ? 18 : 31,
    wellness_score_self_report: band === "High" ? 4 : band === "Moderate" ? 6 : 8,
    sleep_quality_score_self_report: band === "High" ? 3 : band === "Moderate" ? 6 : 8,
    resting_hr_trend_biometric: band === "High" ? 1.8 : band === "Moderate" ? 0.7 : 0.1,
    sleep_hours_biometric: band === "High" ? 5 : band === "Moderate" ? 6.2 : 7.4,
  }));
  return {
    id,
    name,
    service_number: id.replace("demo-person-", "SYN-"),
    rank: "Demo",
    unit: "Synthetic Unit",
    posting: "Demo posting",
    is_demo_data: true,
    features: {},
    raw_records,
    summary: {
      workload,
      wellness,
      leave: "Synthetic sample value",
      deployment: "Synthetic demo posting",
    },
    history: [
      { assessed_at: "2026-10-20T00:00:00Z", band: band === "High" ? "Moderate" : band, probability: Math.max(0.05, probability - 0.08), trust_score: 95, fusion_state: "MONITOR" },
      { assessed_at: "2026-10-27T00:00:00Z", band, probability, trust_score: 95, fusion_state: band === "High" ? "REVIEW_RECOMMENDED" : "MONITOR" },
    ],
  };
}

function staticDemoResponse<T>(method: string, path: string, body?: JsonBody): T {
  if (path === "/auth/login" && method === "POST") {
    const login = (body ?? {}) as { username?: string; demo_role?: string };
    const roles: Record<string, string> = {
      PERSONNEL: "PERSONNEL",
      WELFARE_OFFICER: "WELFARE_OFFICER",
      COMMANDER: "COMMANDER",
    };
    const role = roles[login.demo_role ?? ""];
    if (!role) throw new ApiError(400, { detail: "Choose a demo workspace first." });
    return {
      access_token: `static-demo-${role.toLowerCase()}`,
      token_type: "bearer",
      user_id: `static-${role.toLowerCase()}`,
      username: login.username?.trim() || "demo",
      role,
    } as T;
  }
  if (path === "/auth/logout" && method === "POST") return undefined as T;
  if (path === "/demo/seed" && method === "POST") {
    return { seeded_count: demoPeople.length, message: "Synthetic browser demo is ready." } as T;
  }
  if (path === "/demo/personnel" && method === "GET") return demoPeople as T;
  if (path === "/overview" && method === "GET") {
    return { personnel_count: demoPeople.length, assessments_today: demoPeople.length, open_interventions: 0, high_risk_latest: 1 } as T;
  }
  throw new ApiError(503, { detail: "This feature needs the Manobal-AI backend. The deployed static demo currently includes login and the synthetic officer dashboard." });
}

type JsonBody = unknown;

async function request<T>(method: string, path: string, body?: JsonBody): Promise<T> {
  if (IS_STATIC_DEMO) return staticDemoResponse<T>(method, path, body);
  // Build headers with JWT token if available
  const headers: Record<string, string> = {};
  
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
  }
  
  const token = getToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  // FastAPI reports request-validation failures as 422 with a {detail: [...]} body.
  if (!res.ok) {
    const errBody = await res.json().catch(() => null);
    throw new ApiError(res.status, errBody);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// The response type is yours to declare: nothing infers across the Python boundary, so a
// TS interface here mirrors the endpoint's Pydantic model by hand — keep the two in sync.
export const apiGet = <T>(path: string) => request<T>("GET", path);
export const apiPost = <T>(path: string, body?: JsonBody) => request<T>("POST", path, body ?? null);
export const apiPut = <T>(path: string, body?: JsonBody) => request<T>("PUT", path, body ?? null);
export const apiPatch = <T>(path: string, body?: JsonBody) =>
  request<T>("PATCH", path, body ?? null);
export const apiDelete = <T>(path: string) => request<T>("DELETE", path);
