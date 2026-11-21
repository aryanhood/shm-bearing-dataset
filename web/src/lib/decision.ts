// Mirror of bearingscope/decision.py.

export type Status = "SAFE" | "WATCH" | "CRITICAL";

export const CLASS_TITLES = ["Normal", "Inner race fault", "Outer race fault", "Ball fault"];
const SAFE_MIN_NORMAL = 0.8;
const CRITICAL_MIN_FAULT = 0.8;

export interface Decision {
  status: Status;
  classIndex: number;
  title: string;
  confidence: number;
  action: string;
}

export function decide(proba: number[]): Decision {
  let k = 0;
  for (let i = 1; i < proba.length; i++) if ((proba[i] as number) > (proba[k] as number)) k = i;
  const confidence = proba[k] as number;
  const title = CLASS_TITLES[k] as string;
  if (k === 0 && confidence >= SAFE_MIN_NORMAL) {
    return {
      status: "SAFE",
      classIndex: k,
      title,
      confidence,
      action: "No action. Keep monitoring.",
    };
  }
  if (k !== 0 && confidence >= CRITICAL_MIN_FAULT) {
    return {
      status: "CRITICAL",
      classIndex: k,
      title,
      confidence,
      action: `${title} detected. Schedule bearing replacement.`,
    };
  }
  return {
    status: "WATCH",
    classIndex: k,
    title,
    confidence,
    action: "Signal is ambiguous. Re-check at shorter interval.",
  };
}
