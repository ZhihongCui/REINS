# modules/neo4j_writer.py
from neo4j import GraphDatabase
from typing import List, Dict, Any
from dataclasses import dataclass


# =====================================================
# Rule Data Structure Definition
# =====================================================
@dataclass
class Rule:
    rule_id: str
    cause: str
    effect: str
    rule_type: str       # Physical / Behavioral / Scene
    confidence: float
    raw_text: str
    cause_group: str = "Unknown"
    effect_group: str = "Unknown"
    action: str = ""           # New: structured action
    description: str = ""


# =====================================================
# Neo4j Writer Class (with Variable / Rule / Category three-layer structure)
# =====================================================
class Neo4jWriter:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=None):
        import os
        password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"[Neo4jWriter] Connected to {uri} as {user}")

    def close(self):
        self.driver.close()

    # =====================================================
    # Initialization: optional clear + automatic indexing
    # =====================================================
    def initialize(self, clear: bool = False):
        # (Optional) If you have multi-database enabled, specify the database name:
        # with self.driver.session(database="neo4j") as session:
       with self.driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
            print("Cleared entire Neo4j database.")

        # Neo4j 5: include owningConstraint to avoid dropping backing indexes
        rows = session.run("""
            SHOW INDEXES
            YIELD name, type, entityType, labelsOrTypes, properties, owningConstraint
            RETURN name, type, entityType, labelsOrTypes AS labels, properties, owningConstraint
        """).data()

        to_drop = []
        for r in rows:
            labels = r.get("labels") or []
            props  = r.get("properties") or []
            idx_type = (r.get("type") or "").upper()
            owning = r.get("owningConstraint")

            # Drop ONLY non-unique, non-constraint, legacy indexes we don't want
            is_rule_ruleid = ("Rule" in labels) and ("rule_id" in props)
            is_unique = ("UNIQUE" in idx_type)  # btree-unique, etc.

            if is_rule_ruleid:
                # keep anything unique or owned by a constraint
                continue
            if owning:   # backing a constraint → never drop
                continue
            # (Optional) add your own heuristics for other stale indexes here

            # If you do decide to drop a specific name:
            # to_drop.append(r["name"])

        for name in to_drop:
            session.run(f"DROP INDEX {name} IF EXISTS")
            print(f"Dropped legacy index: {name}")

        # Ensure constraints / indexes
        session.run("""
            CREATE CONSTRAINT rule_id_unique IF NOT EXISTS
            FOR (r:Rule) REQUIRE r.rule_id IS UNIQUE
        """)

        index_cmds = [
            "CREATE INDEX IF NOT EXISTS FOR (v:Variable) ON (v.name)",
            "CREATE INDEX IF NOT EXISTS FOR (v:Variable) ON (v.group)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Rule)   ON (r.type)",
            "CREATE INDEX IF NOT EXISTS FOR (r:Rule)   ON (r.action)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Category) ON (c.name)",
            "CREATE INDEX IF NOT EXISTS FOR (g:Group)  ON (g.name)"
        ]
        for cmd in index_cmds:
            session.run(cmd)
        print("Indexes/constraints ensured for Variable, Rule, Category, Group.")


    # =====================================================
    # Write Variable nodes (with hierarchy attributes)
    # =====================================================
    def write_variable(self, name: str, group: str):
        with self.driver.session() as session:
            session.run("""
                MERGE (v:Variable {name:$name})
                ON CREATE SET v.group = $group
                SET v.group = coalesce(v.group, $group)
                // (Optional) To create Group nodes and edges, uncomment the following three lines:
                MERGE (g:Group {name:$group})
                MERGE (v)-[:IN_GROUP]->(g)
            """, name=name, group=group)

    def write_variables(self, variables: Dict[str, List[str]]):
        with self.driver.session() as session:
            for group, names in variables.items():
                for name in names:
                    session.run("""
                        MERGE (v:Variable {name:$name})
                        ON CREATE SET v.group = $group
                        SET v.group = coalesce(v.group, $group)
                        // (Optional) Group nodes and edges
                        MERGE (g:Group {name:$group})
                        MERGE (v)-[:IN_GROUP]->(g)
                    """, name=name, group=group)
        print(f"[Neo4jWriter] Wrote {sum(len(v) for v in variables.values())} Variable nodes.")


    # =====================================================
    # Write Rule (automatically associate Cause/Effect + Category)
    # =====================================================
    def write_rule(self, rule: Rule):
        with self.driver.session() as session:
            session.execute_write(self._create_rule_tx, rule)

    @staticmethod
    def _create_rule_tx(tx, rule: Rule):
        # --- NEW: Action normalization (upper + enum), with raw_text fallback extraction ---
        act = (rule.action or "").strip().upper()
        ACTIONS = {"IDLE", "FASTER", "SLOWER", "LANE_LEFT", "LANE_RIGHT"}
        if act not in ACTIONS and rule.raw_text:
            import re
            m = re.search(r"\b(IDLE|FASTER|SLOWER|LANE_LEFT|LANE_RIGHT)\b", rule.raw_text.upper())
            if m:
                act = m.group(1)
        # Final act can be empty string; but it's recommended not to create action index when empty (existing index is fine)

        params = {
            **rule.__dict__,
            "action_norm": act
        }

        query = """
        // 1) Cause / Effect Variable (only micro variables; group is just for attributes/search labels)
        MERGE (cause:Variable {name:$cause})
        ON CREATE SET cause.group=$cause_group

        MERGE (effect:Variable {name:$effect})
        ON CREATE SET effect.group=$effect_group

        // 2) Rule node: write normalized r.action
        MERGE (r:Rule {rule_id:$rule_id})
        ON CREATE SET r.text=$raw_text,
                        r.type=$rule_type,
                        r.confidence=$confidence,
                        r.description=$description,
                        r.action=$action_norm
        ON MATCH SET  r.text=$raw_text,
                        r.type=$rule_type,
                        r.confidence=$confidence,
                        r.description=$description,
                        r.action=CASE
                            WHEN $action_norm IS NOT NULL AND $action_norm <> ''
                            THEN $action_norm ELSE r.action END

        // 3) Category
        MERGE (cat:Category {name:$rule_type})

        // 4) Relationships (idempotent)
        MERGE (cause)-[c:CAUSES]->(effect)
        SET c.weight = CASE
        WHEN c.weight IS NULL OR $confidence > c.weight THEN $confidence
        ELSE c.weight
        END
        MERGE (r)-[:TRIGGERS]->(cause)
        MERGE (r)-[:DESCRIBES]->(effect)
        MERGE (r)-[:BELONGS_TO]->(cat)
        """
        tx.run(query, **params)

    def write_rules(self, rules: List[Rule]):
        for rule in rules:
            self.write_rule(rule)
        print(f"[Neo4jWriter] Wrote {len(rules)} Rule nodes + Variable/Category links.")

    # =====================================================
    # Write Episode information (associated with Step and Rule)
    # =====================================================
    def write_episode(self, episode_result: Dict[str, Any]):
        """
        Expected episode_result structure (compatible with runner.py):
        {
          "episode_id": int,
          "total_reward": float,
          "steps": int,
          "trajectory": [
            {
              "step": int,
              "reward": float,
              "candidate_actions": ["IDLE","LANE_LEFT",...],
              "chosen_action_name": "LANE_LEFT",
              "top_rules": [
                {"rule_id": "...", "rule_text": "...", "raw_text":"...", "score": 0.96, "action": "LANE_LEFT"}
              ]
            }, ...
          ]
        }
        """
        with self.driver.session() as session:
            ep_id = episode_result.get("episode_id", 0)
            total_reward = float(episode_result.get("total_reward", 0.0))
            steps = int(episode_result.get("steps", 0))

            # Episode node
            session.run("""
                MERGE (e:Episode {id:$ep_id})
                SET e.total_reward=$total_reward, e.steps=$steps
            """, ep_id=ep_id, total_reward=total_reward, steps=steps)

            for step in episode_result.get("trajectory", []):
                step_idx = int(step.get("step", 0))
                step_id = f"{ep_id}_{step_idx}"

                chosen = str(step.get("chosen_action_name", step.get("action", "N/A")))
                cand = step.get("candidate_actions", []) or []
                reward = float(step.get("reward", 0.0))

                # Step node (save candidate actions array and chosen action)
                session.run("""
                    MERGE (s:Step {id:$step_id})
                    SET s.chosen_action=$chosen,
                        s.candidate_actions=$cands,
                        s.reward=$reward
                    MERGE (e:Episode {id:$ep_id})-[:HAS_STEP]->(s)
                """, step_id=step_id, chosen=chosen, cands=cand, reward=reward, ep_id=ep_id)

                # Step → Rule (save score in relationship; prefer rule_id for rule normalization)
                for r in step.get("top_rules", []):
                    rule_id = r.get("rule_id")
                    rule_text = r.get("rule_text") or r.get("raw_text") or ""
                    rule_action = r.get("action") or ""
                    score = float(r.get("score", 0.0))

                    if rule_id:
                        session.run("""
                            MERGE (ru:Rule {rule_id:$rid})
                            ON CREATE SET ru.text = $rtext, ru.action = $ract
                            ON MATCH SET  ru.text = coalesce(ru.text, $rtext),
                                        ru.action = coalesce(ru.action, $ract)
                            MERGE (s:Step {id:$sid})
                            MERGE (s)-[t:TRIGGERED]->(ru)
                            SET t.score = $score
                        """, rid=rule_id, rtext=rule_text, ract=rule_action, sid=step_id, score=score)
                    else:
                        # Text fallback (may have duplicates)
                        session.run("""
                            MERGE (ru:Rule {text:$rtext})
                            ON CREATE SET ru.action = $ract
                            ON MATCH SET  ru.action = coalesce(ru.action, $ract)
                            MERGE (s:Step {id:$sid})
                            MERGE (s)-[t:TRIGGERED]->(ru)
                            SET t.score = $score
                        """, rtext=rule_text, ract=rule_action, sid=step_id, score=score)

            print(f"[Neo4jWriter] Wrote Episode {ep_id} ({steps} steps, reward={total_reward:.2f})")

    # =====================================================
    # Query Interface
    # =====================================================
    def query_all_variables(self) -> List[str]:
        with self.driver.session() as session:
            result = session.run("MATCH (v:Variable) RETURN v.name AS name")
            return [record["name"] for record in result]

    def query_all_rules(self) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            result = session.run("MATCH (r:Rule) RETURN r")
            return [dict(record["r"]) for record in result]

    def query_downstream(self, variable_name: str) -> List[str]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (:Variable {name:$name})-[:CAUSES*1..2]->(v:Variable)
                RETURN DISTINCT v.name AS name
            """, name=variable_name)
            return [record["name"] for record in result]
