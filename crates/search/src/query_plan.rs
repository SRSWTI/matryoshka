#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    FindSymbol,
    FindBehavior,
    EditTarget,
    TraceDependency,
    ArchitectureOverview,
    TestLookup,
    ReadNext,
    General,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestPreference {
    Prefer,
    Penalize,
    Neutral,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    pub mode: SearchMode,
    pub lexical_weight: f32,
    pub semantic_weight: f32,
    pub symbol_weight: f32,
    pub card_weight: f32,
    pub graph_weight: f32,
    pub include_repo_card: bool,
    pub include_folder_cards: bool,
    pub include_graph_neighbors: bool,
    pub test_preference: TestPreference,
}

pub fn plan_query(query: &str) -> QueryPlan {
    let tokens = query_tokens(query);
    let has = |needles: &[&str]| {
        tokens
            .iter()
            .any(|token| needles.iter().any(|needle| token == needle))
    };
    let has_identifier = tokens.iter().any(|token| {
        token.contains('_')
            || token.contains("::")
            || token.chars().any(|ch| ch.is_ascii_uppercase())
    });

    if has(&[
        "architecture",
        "overview",
        "repo",
        "repository",
        "subsystem",
        "subsystems",
        "map",
    ]) {
        return QueryPlan {
            mode: SearchMode::ArchitectureOverview,
            lexical_weight: 0.9,
            semantic_weight: 1.0,
            symbol_weight: 0.4,
            card_weight: 1.45,
            graph_weight: 0.8,
            include_repo_card: true,
            include_folder_cards: true,
            include_graph_neighbors: false,
            test_preference: TestPreference::Penalize,
        };
    }

    if has(&[
        "test", "tests", "testing", "spec", "fixture", "fixtures", "mock", "mocks", "coverage",
    ]) {
        return QueryPlan {
            mode: SearchMode::TestLookup,
            lexical_weight: 1.2,
            semantic_weight: 0.8,
            symbol_weight: 0.9,
            card_weight: 0.9,
            graph_weight: 0.7,
            include_repo_card: false,
            include_folder_cards: false,
            include_graph_neighbors: true,
            test_preference: TestPreference::Prefer,
        };
    }

    if has(&[
        "edit",
        "change",
        "modify",
        "fix",
        "debug",
        "refactor",
        "implement",
        "update",
        "remove",
        "add",
    ]) {
        return QueryPlan {
            mode: SearchMode::EditTarget,
            lexical_weight: 1.1,
            semantic_weight: 0.9,
            symbol_weight: 1.0,
            card_weight: 1.2,
            graph_weight: 1.0,
            include_repo_card: false,
            include_folder_cards: false,
            include_graph_neighbors: true,
            test_preference: TestPreference::Penalize,
        };
    }

    if has(&[
        "dependency",
        "dependencies",
        "depends",
        "dependent",
        "downstream",
        "upstream",
        "breaks",
        "impact",
        "blast",
        "trace",
        "flow",
    ]) {
        return QueryPlan {
            mode: SearchMode::TraceDependency,
            lexical_weight: 1.0,
            semantic_weight: 0.95,
            symbol_weight: 0.9,
            card_weight: 1.05,
            graph_weight: 1.4,
            include_repo_card: false,
            include_folder_cards: true,
            include_graph_neighbors: true,
            test_preference: TestPreference::Neutral,
        };
    }

    if has(&["read", "explain", "understand", "next"])
        || (has(&["before"]) && !has(&["where", "defined", "definition", "usage", "uses"]))
    {
        return QueryPlan {
            mode: SearchMode::ReadNext,
            lexical_weight: 0.95,
            semantic_weight: 1.05,
            symbol_weight: 0.8,
            card_weight: 1.25,
            graph_weight: 1.1,
            include_repo_card: false,
            include_folder_cards: true,
            include_graph_neighbors: true,
            test_preference: TestPreference::Neutral,
        };
    }

    if has_identifier || has(&["symbol", "defined", "definition", "usage", "uses", "where"]) {
        return QueryPlan {
            mode: SearchMode::FindSymbol,
            lexical_weight: 1.35,
            semantic_weight: 0.7,
            symbol_weight: 1.5,
            card_weight: 0.8,
            graph_weight: 0.8,
            include_repo_card: false,
            include_folder_cards: false,
            include_graph_neighbors: has(&["usage", "uses"]),
            test_preference: TestPreference::Neutral,
        };
    }

    if has(&["behavior", "logic", "responsibility", "handles", "how"]) {
        return QueryPlan {
            mode: SearchMode::FindBehavior,
            lexical_weight: 0.9,
            semantic_weight: 1.2,
            symbol_weight: 0.7,
            card_weight: 1.25,
            graph_weight: 0.8,
            include_repo_card: false,
            include_folder_cards: true,
            include_graph_neighbors: false,
            test_preference: TestPreference::Penalize,
        };
    }

    QueryPlan {
        mode: SearchMode::General,
        lexical_weight: 1.0,
        semantic_weight: 1.0,
        symbol_weight: 1.0,
        card_weight: 1.0,
        graph_weight: 0.8,
        include_repo_card: false,
        include_folder_cards: false,
        include_graph_neighbors: false,
        test_preference: TestPreference::Penalize,
    }
}

fn query_tokens(query: &str) -> Vec<String> {
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != ':')
        .map(str::trim)
        .filter(|token| token.len() > 2)
        .map(ToString::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_symbol_queries_for_exact_retrieval() {
        let plan = plan_query("where is resolve_import defined");
        assert_eq!(plan.mode, SearchMode::FindSymbol);
        assert!(plan.lexical_weight > plan.semantic_weight);
        assert!(plan.symbol_weight > 1.0);
    }

    #[test]
    fn keeps_where_queries_file_or_symbol_oriented_even_with_before() {
        let plan = plan_query("where is advisor called before implementation");
        assert_eq!(plan.mode, SearchMode::FindSymbol);
        assert!(!plan.include_repo_card);
        assert!(!plan.include_folder_cards);
    }

    #[test]
    fn plans_architecture_queries_for_cards() {
        let plan = plan_query("explain repository architecture");
        assert_eq!(plan.mode, SearchMode::ArchitectureOverview);
        assert!(plan.include_repo_card);
        assert!(plan.include_folder_cards);
    }

    #[test]
    fn plans_test_queries_without_test_penalty() {
        let plan = plan_query("tests for parser fixtures");
        assert_eq!(plan.mode, SearchMode::TestLookup);
        assert_eq!(plan.test_preference, TestPreference::Prefer);
    }
}
