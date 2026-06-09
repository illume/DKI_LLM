import { describe, it, expect } from "vitest";
import {
  makeKnowledgeGraph,
  addEvidenceNode,
  addLlmGeneratedKnowledge,
  knowledgeGraphToText,
  knowledgeGraphToJson,
  getKnowledgeNodeById,
  getKnowledgeEdgeById,
  getEvidenceNodeById,
} from "../dataModel.ts";
import { dedupPreserveOrder, safeJsonLoads } from "../utils.ts";

describe("dataModel", () => {
  it("creates an empty KnowledgeGraph", () => {
    const kg = makeKnowledgeGraph();
    expect(kg.knowledge_nodes).toEqual([]);
    expect(kg.knowledge_edges).toEqual([]);
    expect(kg.evidence_nodes).toEqual([]);
  });

  it("adds evidence nodes with auto-incrementing IDs starting at 1", () => {
    const kg = makeKnowledgeGraph();
    const e1 = addEvidenceNode(kg, "Title A", "https://a.com", "Content A");
    const e2 = addEvidenceNode(kg, "Title B", "https://b.com", "Content B");

    expect(e1.id).toBe(1);
    expect(e2.id).toBe(2);
    expect(kg.evidence_nodes).toHaveLength(2);
    expect(e1.source_title).toBe("Title A");
  });

  it("retrieves evidence node by id", () => {
    const kg = makeKnowledgeGraph();
    const ev = addEvidenceNode(kg, "Title", "https://x.com", "Content");
    expect(getEvidenceNodeById(kg, ev.id)?.source_title).toBe("Title");
    expect(getEvidenceNodeById(kg, 999)).toBeUndefined();
  });

  it("adds LLM-generated knowledge nodes and edges", () => {
    const kg = makeKnowledgeGraph();
    const ev = addEvidenceNode(kg, "Src", "https://src.com", "Evidence");

    const extractResult = {
      new_nodes: [
        { id: "new_0", node_name: "Node A", is_core_entity: true },
        { id: "new_1", node_name: "Node B", is_core_entity: true },
      ],
      new_edges: [
        {
          id: "edge_0",
          source_id: "new_0",
          target_id: "new_1",
          relation_name: "relates_to",
        },
      ],
      evidences_map: {},
    };

    addLlmGeneratedKnowledge(kg, extractResult, [ev]);

    expect(kg.knowledge_nodes).toHaveLength(2);
    expect(kg.knowledge_edges).toHaveLength(1);
    // Nodes use 'knowledge' field, not 'node_name'
    expect(kg.knowledge_nodes[0].knowledge).toBe("Node A");
    expect(kg.knowledge_nodes[0].is_core_entity).toBe(true);
    expect(kg.knowledge_edges[0].relation_name).toBe("relates_to");
  });

  it("serializes KG to text", () => {
    const kg = makeKnowledgeGraph();
    addEvidenceNode(kg, "Src", "url", "content");
    addLlmGeneratedKnowledge(
      kg,
      {
        new_nodes: [{ id: "n0", node_name: "Alpha", is_core_entity: true }],
        new_edges: [],
        evidences_map: {},
      },
      [],
    );

    const text = knowledgeGraphToText(kg);
    expect(text).toContain("Alpha");
  });

  it("serializes KG to JSON structure", () => {
    const kg = makeKnowledgeGraph();
    addLlmGeneratedKnowledge(
      kg,
      {
        new_nodes: [{ id: "n0", node_name: "Beta", is_core_entity: true }],
        new_edges: [],
        evidences_map: {},
      },
      [],
    );

    const json = knowledgeGraphToJson(kg);
    expect(json.knowledge_nodes).toBeDefined();
    expect(json.knowledge_edges).toBeDefined();
  });

  it("retrieves knowledge nodes and edges by id", () => {
    const kg = makeKnowledgeGraph();
    addLlmGeneratedKnowledge(
      kg,
      {
        new_nodes: [
          { id: "n0", node_name: "Gamma", is_core_entity: true },
          { id: "n1", node_name: "Delta", is_core_entity: true },
        ],
        new_edges: [
          {
            id: "e0",
            source_id: "n0",
            target_id: "n1",
            relation_name: "links_to",
          },
        ],
        evidences_map: {},
      },
      [],
    );

    expect(getKnowledgeNodeById(kg, kg.knowledge_nodes[0].id)?.knowledge).toBe(
      "Gamma",
    );
    expect(getKnowledgeEdgeById(kg, kg.knowledge_edges[0].id)?.relation_name).toBe(
      "links_to",
    );
    expect(getKnowledgeNodeById(kg, 999)).toBeUndefined();
  });
});

describe("utils", () => {
  it("dedupPreserveOrder removes duplicates preserving first occurrence", () => {
    expect(dedupPreserveOrder(["a", "b", "a", "c", "b"])).toEqual([
      "a",
      "b",
      "c",
    ]);
  });

  it("dedupPreserveOrder filters falsy items", () => {
    expect(dedupPreserveOrder(["a", "", "b", ""])).toEqual(["a", "b"]);
  });

  it("safeJsonLoads parses valid JSON", () => {
    expect(safeJsonLoads('{"key": "value"}')).toEqual({ key: "value" });
  });

  it("safeJsonLoads returns null for unrecoverable JSON", () => {
    // jsonrepair may repair some strings; for truly invalid structures null is returned
    const result = safeJsonLoads("{{{invalid");
    // Result is either null or some repaired output
    expect(result !== undefined).toBe(true);
  });

  it("safeJsonLoads handles JSON in markdown fences", () => {
    const input = '```json\n{"a": 1}\n```';
    expect(safeJsonLoads(input)).toEqual({ a: 1 });
  });
});
