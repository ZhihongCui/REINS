# modules/llm_rule_retriever.py
import json
import re
from typing import Any, Dict, List, Tuple, Optional

from neo4j import GraphDatabase
from src.llm.caller import call_gpt

READ_ONLY_BLOCKLIST = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|LOAD|CALL\s+db\.ms\b|DROP|ALTER)\b",
    flags=re.IGNORECASE
)

SAFE_WHITELIST = re.compile(
    r"^(MATCH|WITH|RETURN|UNWIND|ORDER BY|LIMIT|WHERE|OPTIONAL MATCH|CALL\s+db\.index|CALL\s+db\.schema).*",
    flags=re.IGNORECASE | re.DOTALL
)

# ---- Graph Schema Hint ----
SCHEMA_HINT = """
Graph schema (read-only):
(:Variable {name, group, level})
(:Rule {rule_id, text, type, confidence, description, action})
(:Category {name})

Relationships:
(:Variable)-[:CAUSES {weight}]->(:Variable)
(:Rule)-[:TRIGGERS]->(:Variable)   // cause variable
(:Rule)-[:DESCRIBES]->(:Variable)  // effect variable (textual/semantic)
(:Rule)-[:AFFECTS]->(:Variable)    // variables directly affected by the rule (operational)
(:Rule)-[:BELONGS_TO]->(:Category)
"""



# ---- Fallback Query ----
FALLBACK_CYPHER = """
MATCH (r:Rule)-[:TRIGGERS]->(c:Variable)
OPTIONAL MATCH (c)-[:CAUSES*0..1]->(e:Variable)
WITH r, c, collect(DISTINCT e) AS Elist
OPTIONAL MATCH (r)-[:AFFECTS]->(ev:Variable)
WITH r, c, Elist, collect(DISTINCT ev) AS EVlist
WHERE c.name IN $active_causes
  AND (
    any(e IN Elist WHERE e.group IN $target_groups) OR
    any(ev IN EVlist WHERE ev.group IN $target_groups)
  )
RETURN
  r.rule_id AS rule_id,
  r.text AS raw_text,
  r.action AS action,
  c.name AS cause,
  [e IN Elist  | e.name] AS effects,
  [v IN EVlist | v.name] AS effect_vars,
  r.type AS rule_type,
  coalesce(r.confidence, 0.0) AS confidence
ORDER BY confidence DESC
LIMIT $k
"""


def summarize_state_for_prompt(sample: Dict[str, Any], max_keys_per_group: int = 6) -> Dict[str, Any]:
    """Compress the current state into an LLM-readable summary: top-K key variables and values for each group."""
    out = {}
    for group in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
        g = sample.get(group, {}) or {}
        items = list(g.items())[:max_keys_per_group]
        out[group] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in items}
    return out

def active_variables_heuristic(sample: Dict[str, Any], thr: float = 1e-6) -> List[str]:
    """Active-variable selector: numeric keys with non-zero magnitude form the query set."""
    act = []
    for group in ["Environment", "Traffic", "Ego", "Action", "Outcome"]:
        g = sample.get(group, {}) or {}
        for k, v in g.items():
            if isinstance(v, (int, float)) and abs(float(v)) > thr:
                act.append(k)
    # Remove duplicates
    seen, uniq = set(), []
    for k in act:
        if k not in seen:
            seen.add(k); uniq.append(k)
    return uniq


class LLMRuleRetriever:
    """
    Use LLM to generate 'read-only Cypher' queries to dynamically retrieve rules related to the current state from Neo4j.
    Features: safety check (read-only), parameterized execution, fallback mechanism, optional debug output.
    """

    def __init__(self,
                 uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = None,  # Set via NEO4J_PASSWORD env var or pass explicitly
                 llm_model: str = "gpt-4.1-nano",
                 max_rules: int = 20,
                 debug: bool = False):
        import os
        password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.llm_model = llm_model
        self.max_rules = max_rules
        self.debug = debug
        self._last_debug: Dict[str, Any] = {}  # Store the most recent debug info

    # External access to the most recent debug info
    def get_last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    def close(self):
        self.driver.close()

    # ------------------ Main Entry Point ------------------
    def retrieve(self, sample: Dict[str, Any], k: int = 10,
                 target_groups: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Returns List[rule dict] with fields including:
        raw_text,
        "action": str,       # Structured action (e.g., "LANE_LEFT"/"FASTER"...)
        "cause": str,        # Trigger variable name (may be empty)
        cause,
        effect,
        rule_type,
        confidence,
        rule_id
        """
        k = min(k, self.max_rules)
        target_groups = target_groups or ["Outcome", "Ego", "Action"]

        state_summary = summarize_state_for_prompt(sample)
        active_vars = active_variables_heuristic(sample)

        # Initialize last_debug
        self._last_debug = {
            "state_summary": state_summary,
            "active_vars": active_vars,
            "target_groups": target_groups,
            "k": k,
            "prompt": None,
            "cypher": None,
            "params": None,
            "fallback_used": False,
            "rows_preview": None,
            "error": None,
        }

        # 1) Let LLM generate parameterized Cypher (JSON wrapped)
        cypher, params = self._llm_cypher(state_summary, target_groups, k)
        self._last_debug["cypher"] = cypher
        self._last_debug["params"] = params

        if self.debug:
            print("\n[LLMRuleRetriever/DEBUG] --- LLM Cypher PROMPT ---")
            print(self._last_debug.get("prompt") or "")
            print("\n[LLMRuleRetriever/DEBUG] --- LLM Cypher OUTPUT ---")
            print("Cypher:", cypher or "<None>")
            print("Params:", json.dumps(params or {}, ensure_ascii=False, indent=2))

        # 2) Safety check; fallback if failed
        if (not self._is_safe_cypher(cypher)) or (READ_ONLY_BLOCKLIST.search(cypher or "")):
            if self.debug:
                print("[LLMRuleRetriever/DEBUG] Cypher unsafe or blocked. Falling back.")
            cypher = None

        # 3) Execute LLM-generated query
        try:
            if cypher:
                rows = self._run_read(cypher, params)
                if rows:
                    if self.debug:
                        print(f"[LLMRuleRetriever/DEBUG] Retrieved {len(rows)} rules (LLM Cypher).")
                        for i, r in enumerate(rows[:10]):
                            print(f"  #{i+1}: rule_id={r.get('rule_id')}  conf={r.get('confidence'):.3f} "
                                f"cause={r.get('cause')}  effects={r.get('effects')}  effect_vars={r.get('effect_vars')}\n"
                                f"     text={r.get('raw_text')}")

                    self._last_debug["rows_preview"] = rows[:10]
                    return rows
        except Exception as e:
            msg = f"LLM cypher failed: {e}"
            self._last_debug["error"] = msg
            if self.debug:
                print(f"[LLMRuleRetriever/DEBUG] {msg}")

        # 4) Fallback query (stable)
        try:
            fb_params = {
                "active_causes": active_vars[:50],
                "target_groups": target_groups,
                "k": k
            }
            rows = self._run_read(FALLBACK_CYPHER, fb_params)
            self._last_debug["fallback_used"] = True
            self._last_debug["cypher"] = FALLBACK_CYPHER
            self._last_debug["params"] = fb_params
            if self.debug:
                print(f"[LLMRuleRetriever/DEBUG] Fallback query used. Retrieved {len(rows)} rules.")
                for i, r in enumerate(rows[:10]):
                    print(f"  (FB) #{i+1}: rule_id={r.get('rule_id')}  conf={r.get('confidence'):.3f}  "
                          f"cause={r.get('cause')}  effect={r.get('effect')}\n       text={r.get('raw_text')}")
            self._last_debug["rows_preview"] = rows[:10]
            return rows
        except Exception as e:
            msg = f"Fallback cypher failed: {e}"
            self._last_debug["error"] = msg
            if self.debug:
                print(f"[LLMRuleRetriever/DEBUG] {msg}")
            return []

    # ------------------ LLM Generate Parameterized Cypher ------------------
    def _llm_cypher(self, state_summary: Dict[str, Any],
                    target_groups: List[str], k: int) -> Tuple[Optional[str], Dict[str, Any]]:
        prompt = f"""
You act as a Cypher generator to retrieve driving rules relevant to the current state.
Return STRICT JSON with keys: "cypher", "params".
- The Cypher MUST be READ-ONLY (MATCH/RETURN/WHERE/ORDER BY/LIMIT ...).
- MUST be parameterized (NO hard-coded literals for lists/numbers; use $params).
- Target rule pattern:
  (r:Rule)-[:TRIGGERS]->(c:Variable)
  OPTIONAL (c)-[:CAUSES*0..1]->(e:Variable)
  OPTIONAL (r)-[:AFFECTS]->(ev:Variable)
  At least one of e.group or ev.group must be in $target_groups (usually Ego/Outcome/Traffic).
- Prefer higher r.confidence; keep LIMIT $k.

{SCHEMA_HINT}

Current state summary (non-exhaustive):
{json.dumps(state_summary, indent=2, ensure_ascii=False)}

Target groups to describe: {target_groups}
Top-k to return: {k}

Example expected JSON:
{{
  "cypher": "MATCH (r:Rule)-[:TRIGGERS]->(c:Variable) OPTIONAL MATCH (c)-[:CAUSES*0..1]->(e:Variable) WITH r, c, collect(DISTINCT e) AS Elist OPTIONAL MATCH (r)-[:AFFECTS]->(ev:Variable) WITH r, c, Elist, collect(DISTINCT ev) AS EVlist WHERE c.name IN $active_causes AND ( any(e IN Elist WHERE e.group IN $target_groups) OR any(ev IN EVlist WHERE ev.group IN $target_groups) ) RETURN r.rule_id AS rule_id, r.text AS raw_text, r.action AS action, c.name AS cause, [e IN Elist | e.name] AS effects, [v IN EVlist | v.name] AS effect_vars, r.type AS rule_type, coalesce(r.confidence, 0.0) AS confidence ORDER BY confidence DESC LIMIT $k",
  "params": {{
     "active_causes": ["headway_distance","ego_speed"],
     "target_groups": ["Outcome", "Ego", "Traffic"],
     "k": 10
  }}
}}
Only output the JSON. No commentary.
"""

        # Record prompt in debug info
        self._last_debug["prompt"] = prompt

        try:
            ans = call_gpt(prompt, model=self.llm_model, temperature=0.2, max_tokens=600)
            js = json.loads(ans)
            cypher = js.get("cypher", "")
            params = js.get("params", {})
            # Fallback: provide params if LLM didn't fill them
            params.setdefault("target_groups", target_groups)
            params.setdefault("k", k)
            # Defense: keep active_causes as empty list when not provided (upper layer can replace with heuristic)
            if not params.get("active_causes"):
                params["active_causes"] = []
            return cypher, params
        except Exception as e:
            if self.debug:
                print(f"[LLMRuleRetriever/DEBUG] LLM parse failed: {e}")
            return None, {}

    # ------------------ Neo4j READ Execution ------------------
    def _run_read(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        def work(tx):
            res = tx.run(cypher, **params)
            out = []
            for r in res:
                out.append({
                    "rule_id": r.get("rule_id"),
                    "raw_text": r.get("raw_text") or "",
                    "action": r.get("action") or "",
                    "cause": r.get("cause") or "",
                    "effects": r.get("effects") or [],
                    "effect_vars": r.get("effect_vars") or [],
                    "rule_type": r.get("rule_type") or "",
                    "confidence": float(r.get("confidence") or 0.0),
                })

            return out
        with self.driver.session() as sess:
            return sess.execute_read(work)



    # ------------------ Safety Check ------------------
    def _is_safe_cypher(self, q: Optional[str]) -> bool:
        if not q:
            return False
        if READ_ONLY_BLOCKLIST.search(q):
            return False
        # Simple whitelist: first word must be read-only syntax
        first = q.strip().split(None, 1)[0].upper()
        return first in ("MATCH", "WITH", "OPTIONAL", "CALL", "UNWIND", "RETURN")
